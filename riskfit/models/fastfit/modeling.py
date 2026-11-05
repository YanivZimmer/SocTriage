# Vendored and modified copy of IBM FastFit (https://github.com/IBM/fastfit, Apache License 2.0).
# Modifications: LoRA-friendly encoder handling, Apple-Silicon (MPS) fixes, a classification-head
# loss path with optional focal-Tversky, soft-recall hinge (RiskFit), soft false-positive penalty and
# reward terms, and a malicious/non-malicious balanced batch sampler. See riskfit/models/fastfit_lora.py
# for the configuration actually used in the paper.
import copy
import json
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

try:
    from torch.nn.attention import SDPBackend
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel

    def _math_sdpa_ctx():
        return _sdpa_kernel([SDPBackend.MATH])

except ImportError:
    # Older PyTorch builds that don't have torch.nn.attention
    def _math_sdpa_ctx():
        return nullcontext()


from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(
        self,
        temperature=0.07,
        contrast_mode="all",
        base_temperature=0.07,
        contrastive_variant="standard",
        target_label_id=None,
        target_positive_margin=0.0,
        target_anchor_weight=1.0,
        target_hard_negative_strength=0.0,
        target_hard_positive_strength=0.0,
        target_logit_scale=1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.contrastive_variant = contrastive_variant
        self.target_label_id = target_label_id
        self.target_positive_margin = target_positive_margin
        self.target_anchor_weight = target_anchor_weight
        self.target_hard_negative_strength = target_hard_negative_strength
        self.target_hard_positive_strength = target_hard_positive_strength
        self.target_logit_scale = target_logit_scale

    def forward(self, features, labels=None, mask=None, sim=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # FIX: Add MPS device support for Apple Silicon GPU
        if features.is_cuda:
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and features.device.type == "mps":
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...],at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.long().contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # compute logits
        if sim is None:
            anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        else:
            anchor_dot_contrast = sim / self.temperature

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask
        flat_labels = labels.contiguous().view(-1).to(device) if labels is not None else None

        if self.target_label_id is not None and self.target_logit_scale != 1.0 and flat_labels is not None:
            anchor_labels = flat_labels.repeat(anchor_count)
            target_anchor_mask = (anchor_labels == self.target_label_id).float().view(-1, 1)
            logits = logits * torch.where(
                target_anchor_mask > 0,
                torch.full_like(logits, self.target_logit_scale),
                torch.ones_like(logits),
            )

        if self.target_label_id is not None and self.target_positive_margin > 0.0 and flat_labels is not None:
            anchor_labels = flat_labels.repeat(anchor_count)
            target_anchor_mask = (anchor_labels == self.target_label_id).float().view(-1, 1)
            logits = logits + (self.target_positive_margin * target_anchor_mask * mask)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        if self.target_label_id is not None and self.target_hard_negative_strength > 0.0 and flat_labels is not None:
            anchor_labels = flat_labels.repeat(anchor_count)
            target_anchor_mask = (anchor_labels == self.target_label_id).float().view(-1, 1)
            negative_mask = (1.0 - mask) * logits_mask
            hard_weights = 1.0 + self.target_hard_negative_strength * torch.sigmoid(logits.detach())
            exp_logits = exp_logits * torch.where(
                (target_anchor_mask * negative_mask) > 0,
                hard_weights,
                torch.ones_like(hard_weights),
            )
        if self.contrastive_variant == "class_balanced" and labels is not None:
            contrast_labels = flat_labels.repeat(contrast_count)
            class_balanced_denominator = torch.zeros(
                (batch_size * anchor_count, 1),
                dtype=exp_logits.dtype,
                device=device,
            )
            for class_id in torch.unique(contrast_labels):
                class_mask = (contrast_labels == class_id).float().view(1, -1)
                class_logits_mask = logits_mask * class_mask
                class_count = class_logits_mask.sum(1, keepdim=True).clamp_min(1.0)
                class_balanced_denominator += (exp_logits * class_mask).sum(1, keepdim=True) / class_count
            log_prob = logits - torch.log(class_balanced_denominator + 1e-6)
        else:
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        if self.target_label_id is not None and self.target_hard_positive_strength > 0.0 and flat_labels is not None:
            anchor_labels = flat_labels.repeat(anchor_count)
            target_anchor_mask = (anchor_labels == self.target_label_id).float().view(-1, 1)
            hard_positive_weights = 1.0 + self.target_hard_positive_strength * torch.sigmoid(-logits.detach())
            positive_weights = mask * torch.where(
                target_anchor_mask > 0,
                hard_positive_weights,
                torch.ones_like(hard_positive_weights),
            )
            mean_log_prob_pos = (positive_weights * log_prob).sum(1) / (positive_weights.sum(1) + 1e-6)
        else:
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        if self.target_label_id is not None and self.target_anchor_weight != 1.0 and flat_labels is not None:
            sample_weights = torch.where(
                flat_labels == self.target_label_id,
                torch.full_like(flat_labels, self.target_anchor_weight, dtype=loss.dtype),
                torch.ones_like(flat_labels, dtype=loss.dtype),
            )
            loss = (loss.view(anchor_count, batch_size) * sample_weights.view(1, -1)).sum()
            loss = loss / (sample_weights.sum() * anchor_count).clamp_min(1e-6)
        else:
            loss = loss.view(anchor_count, batch_size).mean()

        return loss


class LabelAnchorContrastiveLoss(nn.Module):
    """Use batch class prototypes as label anchors in embedding space."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        if labels is None:
            return features.new_tensor(0.0)

        labels = labels.long().contiguous().view(-1)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_labels = labels.repeat(features.shape[1])

        unique_labels = torch.unique(labels)
        if unique_labels.numel() < 2:
            return features.new_tensor(0.0)

        prototypes = []
        for label in unique_labels:
            prototypes.append(contrast_feature[contrast_labels == label].mean(dim=0))
        prototypes = torch.stack(prototypes, dim=0)

        logits = torch.matmul(contrast_feature, prototypes.T) / self.temperature
        targets = torch.zeros_like(contrast_labels, dtype=torch.long)
        for idx, label in enumerate(unique_labels):
            targets = torch.where(contrast_labels == label, idx, targets)

        return F.cross_entropy(logits, targets)


def set_dropout(model, dropout):
    def _set_dropout(module, dropout):
        if isinstance(module, torch.nn.modules.dropout.Dropout):
            module.p = dropout
        else:
            if hasattr(module, "dropout") and isinstance(module.dropout, float):
                module.dropout = dropout
            if hasattr(module, "activation_dropout") and isinstance(module.activation_dropout, float):
                module.activation_dropout = dropout

    _set_dropout(model, dropout)
    for module in model.modules():
        _set_dropout(module, dropout)


@dataclass
class ConfigArguments:
    num_repeats: int | None = field(
        default=1,
        metadata={"help": "The number of times to repeat the queries and docs in every batch."},
    )
    proj_dim: int | None = field(default=128, metadata={"help": "The dimension of the projection layer."})
    clf_loss_factor: float | None = field(
        default=0.1, metadata={"help": "The factor to scale the classification loss."}
    )
    sim_factor: float | None = field(default=1.0, metadata={"help": "The factor to scale the similarity loss."})
    pretrain_mode: bool | None = field(default=False, metadata={"help": "Whether to do pre-training."})
    inference_type: str | None = field(default="sim", metadata={"help": "The inference type to be used."})
    pooled_cosine_scale: float | None = field(
        default=1.0,
        metadata={"help": "Scale applied to pooled-cosine logits."},
    )
    pooled_cosine_learn_scale: bool | None = field(
        default=False,
        metadata={"help": "Whether to learn the pooled-cosine logit scale."},
    )
    inference_direction: str | None = field(
        default="doc", metadata={"help": "The inference direction to be used from doc/query/both."}
    )
    rep_tokens: str | None = field(
        default="all",
        metadata={
            "help": "The tokens to use for representation when calculating the similarity in training and inference."
        },
    )
    length_norm: bool | None = field(
        default=True,
        metadata={"help": "Whether to normalize by length while considering pad"},
    )
    length_norm_at_inference: bool | None = field(
        default=False,
        metadata={"help": "Whether to normalize by length at inference"},
    )
    mlm_factor: float | None = field(default=0.0, metadata={"help": "The factor to scale the MLM loss."})
    mask_prob: float | None = field(default=0.0, metadata={"help": "The probability of masking a token."})
    clf_level: str | None = field(
        default="cls", metadata={"help": "The level in which to apply classifier loss on from [mean_pool, cls, token]"}
    )
    mask_zeros_in_query: bool | None = field(
        default=False,
        metadata={"help": "Whether to put mask token on every zero token in every query."},
    )
    mask_zeros_in_doc: bool | None = field(
        default=False,
        metadata={"help": "Whether to put mask token on every zero token in every doc."},
    )
    symetric_mode: bool | None = field(
        default=True,
        metadata={"help": "Whether to compare the doc and query in a symetric mode"},
    )
    scores_temp: float | None = field(
        default=0.07,
        metadata={"help": "Temperature for the SupConLoss similarity scoring. Lower = sharper contrastive boundaries."},
    )
    contrastive_variant: str | None = field(
        default="standard",
        metadata={"help": "Contrastive loss variant: standard or class_balanced."},
    )
    label_anchor_factor: float | None = field(
        default=0.0,
        metadata={"help": "Weight for auxiliary batch label-prototype contrastive loss."},
    )
    target_label_id: int | None = field(
        default=None,
        metadata={"help": "Optional numeric label id for target-class contrastive emphasis."},
    )
    target_positive_margin: float | None = field(
        default=0.0,
        metadata={"help": "Extra logit margin applied to target-class positive contrastive pairs."},
    )
    target_anchor_weight: float | None = field(
        default=1.0,
        metadata={"help": "Loss weight for anchors from the target class."},
    )
    target_hard_negative_strength: float | None = field(
        default=0.0,
        metadata={"help": "Extra hard-negative weighting applied only to target-class anchors."},
    )
    target_hard_positive_strength: float | None = field(
        default=0.0,
        metadata={"help": "Extra hard-positive weighting applied only to target-class anchors."},
    )
    target_logit_scale: float | None = field(
        default=1.0,
        metadata={"help": "Class-conditional logit scale applied to target-class anchors."},
    )
    tversky_loss_weight: float | None = field(
        default=0.0,
        metadata={"help": "Weight for target-class Tversky loss. 0 disables it."},
    )
    tversky_alpha: float | None = field(
        default=0.3,
        metadata={"help": "Tversky penalty on soft false positives for the target class."},
    )
    tversky_beta: float | None = field(
        default=0.7,
        metadata={"help": "Tversky penalty on soft false negatives for the target class."},
    )
    tversky_gamma: float | None = field(
        default=1.0,
        metadata={"help": "Focal exponent applied to (1 - Tversky index)."},
    )
    recall_constraint_rho: float | None = field(
        default=0.0,
        metadata={"help": "Target soft recall threshold for the rare class. Hinge activates below this."},
    )
    recall_constraint_lambda: float | None = field(
        default=0.0,
        metadata={"help": "Lagrangian penalty weight for falling below recall_constraint_rho."},
    )
    recall_hinge_window: str | None = field(
        default="micro",
        metadata={"help": "'micro' = soft recall per micro-batch (the paper's recipe); 'accum' = soft recall over the "
                          "gradient-accumulation window (natural-sampling arms of the hinge-by-sampling experiment)."},
    )
    false_positive_penalty_beta: float | None = field(
        default=0.0,
        metadata={"help": "Penalty weight on mean p(rare class) over non-rare examples."},
    )
    constrained_reward_metric: str | None = field(
        default="none",
        metadata={
            "help": "Reward metric: 'soft_accuracy', 'soft_noise_reduction', 'soft_benign_noise_reduction', or 'none'."
        },
    )
    constrained_reward_weight: float | None = field(
        default=0.0,
        metadata={"help": "Reward weight subtracted from the Lagrangian loss."},
    )
    adaptive_dual_enabled: bool | None = field(
        default=False,
        metadata={"help": "Adapt recall_constraint_lambda during training with clipped dual updates."},
    )
    adaptive_dual_initial_lambda: float | None = field(
        default=0.0,
        metadata={"help": "Initial lambda value when adaptive dual is enabled."},
    )
    adaptive_dual_eta: float | None = field(
        default=0.05,
        metadata={"help": "Dual learning rate (step size for lambda updates)."},
    )
    adaptive_dual_max_lambda: float | None = field(
        default=5.0,
        metadata={"help": "Maximum value for the adaptive dual variable lambda."},
    )
    adaptive_dual_ema_decay: float | None = field(
        default=0.9,
        metadata={"help": "EMA decay for smoothing recall gap before dual update."},
    )
    adaptive_dual_warmup_steps: int | None = field(
        default=0,
        metadata={"help": "Steps before dual updates begin."},
    )
    adaptive_dual_update_interval_steps: int | None = field(
        default=1,
        metadata={"help": "Number of steps between dual variable updates."},
    )


class FastFitConfig(PretrainedConfig):
    model_type = "FastFit"
    is_composition = True

    def __init__(
        self,
        all_docs=None,
        num_repeats=1,
        rep_tokens="all",
        inference_type="sim",
        clf_level="cls",
        init_freeze="all",
        clf_dim=77,
        proj_dim=128,
        similarity_metric="cosine",
        encoder_dropout=None,
        encoding_type="dual",
        clf_query=True,
        clf_doc=False,
        mask_zeros_in_query=False,
        mask_zeros_in_doc=False,
        sim_factor=1.0,
        clf_factor=0.1,
        mlm_factor=0.0,
        mask_prob=0.15,
        pretrain_mode=False,
        length_norm=True,  # wether to noramlize scores by length
        scores_temp=0.07,
        contrastive_variant="standard",
        label_anchor_factor=0.0,
        target_label_id=None,
        target_positive_margin=0.0,
        target_anchor_weight=1.0,
        target_hard_negative_strength=0.0,
        target_hard_positive_strength=0.0,
        target_logit_scale=1.0,
        pooled_cosine_scale=1.0,
        pooled_cosine_learn_scale=False,
        inference_direction="doc",
        symetric_mode=True,
        length_norm_at_inference=False,
        clf_loss_factor=None,
        sim_loss_factor=None,
        tversky_loss_weight=0.0,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        tversky_gamma=1.0,
        recall_constraint_rho=0.0,
        recall_constraint_lambda=0.0,
        recall_hinge_window="micro",
        false_positive_penalty_beta=0.0,
        constrained_reward_metric="none",
        constrained_reward_weight=0.0,
        adaptive_dual_enabled=False,
        adaptive_dual_initial_lambda=0.0,
        adaptive_dual_eta=0.05,
        adaptive_dual_max_lambda=5.0,
        adaptive_dual_ema_decay=0.9,
        adaptive_dual_warmup_steps=0,
        adaptive_dual_update_interval_steps=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.has_no_defaults_at_init = True

        if clf_loss_factor is not None:
            clf_factor = clf_loss_factor
        if sim_loss_factor is not None:
            sim_factor = sim_loss_factor
        pooled_cosine_scale = float(pooled_cosine_scale)
        if pooled_cosine_scale <= 0.0:
            raise ValueError("pooled_cosine_scale must be positive")

        self.all_docs = all_docs
        self.num_repeats = num_repeats
        self.rep_tokens = rep_tokens
        self.clf_level = clf_level
        self.init_freeze = init_freeze
        self.clf_dim = clf_dim
        self.proj_dim = proj_dim
        self.similarity_metric = similarity_metric
        self.encoder_dropout = encoder_dropout
        self.encoding_type = encoding_type
        self.clf_query = clf_query if not pretrain_mode else False
        self.clf_doc = clf_doc
        self.mask_zeros_in_query = mask_zeros_in_query
        self.mask_zeros_in_doc = mask_zeros_in_doc
        self.sim_factor = sim_factor
        self.clf_factor = clf_factor
        self.mlm_factor = mlm_factor
        self.inference_type = inference_type
        self.pretrain_mode = pretrain_mode
        self.length_norm = length_norm
        self.scores_temp = scores_temp
        self.contrastive_variant = contrastive_variant
        self.label_anchor_factor = label_anchor_factor
        self.target_label_id = target_label_id
        self.target_positive_margin = target_positive_margin
        self.target_anchor_weight = target_anchor_weight
        self.target_hard_negative_strength = target_hard_negative_strength
        self.target_hard_positive_strength = target_hard_positive_strength
        self.target_logit_scale = target_logit_scale
        self.tversky_loss_weight = float(tversky_loss_weight or 0.0)
        self.tversky_alpha = float(tversky_alpha or 0.0)
        self.tversky_beta = float(tversky_beta or 0.0)
        self.tversky_gamma = float(tversky_gamma or 1.0)
        self.recall_constraint_rho = float(recall_constraint_rho or 0.0)
        self.recall_constraint_lambda = float(recall_constraint_lambda or 0.0)
        self.recall_hinge_window = str(recall_hinge_window or "micro")
        assert self.recall_hinge_window in ("micro", "accum"), self.recall_hinge_window
        self.false_positive_penalty_beta = float(false_positive_penalty_beta or 0.0)
        self.constrained_reward_metric = constrained_reward_metric or "none"
        self.constrained_reward_weight = float(constrained_reward_weight or 0.0)
        self.adaptive_dual_enabled = bool(adaptive_dual_enabled)
        self.adaptive_dual_initial_lambda = float(adaptive_dual_initial_lambda or 0.0)
        self.adaptive_dual_eta = float(adaptive_dual_eta or 0.0)
        self.adaptive_dual_max_lambda = float(adaptive_dual_max_lambda or 0.0)
        self.adaptive_dual_ema_decay = float(adaptive_dual_ema_decay or 0.0)
        self.adaptive_dual_warmup_steps = int(adaptive_dual_warmup_steps or 0)
        self.adaptive_dual_update_interval_steps = max(1, int(adaptive_dual_update_interval_steps or 1))
        self.pooled_cosine_scale = pooled_cosine_scale
        self.pooled_cosine_learn_scale = pooled_cosine_learn_scale
        self.inference_direction = inference_direction
        self.mlm_prob = mask_prob
        self.mask_prob = mask_prob
        self.symetric_mode = symetric_mode
        self.length_norm_at_inference = length_norm_at_inference

        assert inference_direction in ["query", "doc", "both"]
        if "encoder" in kwargs:
            encoder_config = kwargs.pop("encoder")
            encoder_model_type = encoder_config.pop("model_type")
            self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
            self.hidden_size = self.encoder.hidden_size
        else:
            # transformers introspects config defaults by constructing FastFitConfig()
            # with no args (e.g. _get_non_default_generation_parameters during
            # save_pretrained). Allow that no-encoder construction instead of asserting
            # — real models always supply the encoder via from_encoder_config().
            self.encoder = None
            self.hidden_size = None

    @classmethod
    def from_encoder_config(cls, encoder_config: PretrainedConfig, **kwargs) -> PretrainedConfig:
        return cls(encoder=encoder_config.to_dict(), **kwargs)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["encoder"] = self.encoder.to_dict() if self.encoder is not None else None
        output["model_type"] = self.__class__.model_type
        return output

    @property
    def has_constrained_objective(self) -> bool:
        """Whether an auxiliary constrained objective needs classifier logits."""
        return (
            self.recall_constraint_lambda > 0.0
            or self.false_positive_penalty_beta > 0.0
            or self.constrained_reward_weight > 0.0
            or self.adaptive_dual_enabled
        )

    @property
    def requires_classifier_head(self) -> bool:
        """Whether training must retain the classifier head and its criterion."""
        return self.clf_factor > 0.0 or self.tversky_loss_weight > 0.0 or self.has_constrained_objective


class FastFitTrainable(PreTrainedModel):
    config_class = FastFitConfig
    base_model_prefix = "FastFit"

    def __init__(
        self,
        config: PretrainedConfig | None = None,
        encoder: PreTrainedModel | None = None,
    ):
        if config is None and encoder is None:
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = FastFitConfig.from_encoder_config(encoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        super().__init__(config)

        if encoder is None:
            encoder = AutoModel.from_config(config.encoder)

        self.encoder = encoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )

        self.projection = nn.Linear(config.hidden_size, config.proj_dim, bias=False)
        self.inner_dim = config.hidden_size
        if config.pooled_cosine_learn_scale:
            initial_scale = torch.tensor(float(config.pooled_cosine_scale)).clamp_min(1e-6)
            self.pooled_cosine_log_scale = nn.Parameter(torch.log(initial_scale))

        if config.clf_query or config.requires_classifier_head:
            if config.clf_level in ["cls", "mean_pool"]:
                self.clf = nn.Linear(config.hidden_size, config.clf_dim)

            elif config.clf_level == "token":
                self.clf = nn.Linear(self.inner_dim * config.head_token_size, config.clf_dim)

        self.dropout = nn.Dropout(0.1)  # follow the default in bert model
        self.batch_norm = nn.BatchNorm1d(num_features=self.inner_dim)

        if config.encoder_dropout is not None:
            set_dropout(self.encoder, config.encoder_dropout)

        self.skiplist = {}

        if config.mask_prob > 0.0:
            self.tokenizer = AutoTokenizer.from_pretrained(config.encoder._name_or_path)

        if config.mlm_factor > 0.0:
            self.mlm_criterion = nn.CrossEntropyLoss()
            self.lm_head = AutoModelForMaskedLM.from_pretrained(config.encoder._name_or_path).lm_head

        if config.requires_classifier_head:
            self.clf_criterion = nn.CrossEntropyLoss()
        else:
            self.config.clf_doc = False
            self.config.clf_query = False

        if config.sim_factor > 0.0:
            self.sim_criterion = SupConLoss(
                temperature=config.scores_temp,
                contrastive_variant=config.contrastive_variant,
                target_label_id=config.target_label_id,
                target_positive_margin=config.target_positive_margin,
                target_anchor_weight=config.target_anchor_weight,
                target_hard_negative_strength=config.target_hard_negative_strength,
                target_hard_positive_strength=config.target_hard_positive_strength,
                target_logit_scale=config.target_logit_scale,
            )
            self.label_anchor_criterion = LabelAnchorContrastiveLoss(temperature=config.scores_temp)

        if config.all_docs is not None:
            input_ids, attention_mask = config.all_docs
            input_ids = torch.LongTensor(input_ids)
            attention_mask = torch.LongTensor(attention_mask)
            self.set_documents((input_ids, attention_mask))

        initial_lambda = (
            config.adaptive_dual_initial_lambda if config.adaptive_dual_enabled else config.recall_constraint_lambda
        )
        self.register_buffer(
            "_adaptive_dual_lambda",
            torch.tensor(float(initial_lambda or 0.0), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("_adaptive_dual_gap_ema", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.register_buffer("_adaptive_dual_steps", torch.tensor(0, dtype=torch.long), persistent=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        from_tf = kwargs.pop("from_tf", False)
        if from_tf:
            raise ValueError("Loading a TensorFlow model in PyTorch is not supported.")

        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for EncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def from_encoder_pretrained(
        cls, encoder_pretrained_model_name_or_path: str = None, *model_args, **kwargs
    ) -> PreTrainedModel:
        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path,
                    **kwargs_encoder,
                    return_unused_kwargs=True,
                )

                # Use getattr for compatibility with model configs that don't have these attrs
                is_decoder = getattr(encoder_config, "is_decoder", False)
                add_cross_attention = getattr(encoder_config, "add_cross_attention", False)
                if is_decoder is True or add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    if hasattr(encoder_config, "is_decoder"):
                        encoder_config.is_decoder = False
                    if hasattr(encoder_config, "add_cross_attention"):
                        encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        # instantiate config with corresponding kwargs
        if "config" not in kwargs:
            config = FastFitConfig.from_encoder_config(encoder.config, **kwargs)
        else:
            config = kwargs.pop("config")

        return cls(encoder=encoder, config=config)

    def get_encoder(self):
        return self.encoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def resize_token_embeddings(self, *args, **kwargs):
        embeddings = self.encoder.resize_token_embeddings(*args, **kwargs)
        self.config.encoder.vocab_size = embeddings.num_embeddings
        return embeddings

    def set_documents(self, all_docs):
        input_ids, attention_mask = all_docs
        self.config.all_docs = [input_ids.tolist(), attention_mask.tolist()]
        # Assuming 'all_docs' is your list of torch.LongTensor objects
        all_docs = [torch.nn.Parameter(doc, requires_grad=False) for doc in all_docs]

        # Now you can create a ParameterList
        self.all_docs = torch.nn.ParameterList(all_docs)

    def inference_forward(self, query_input_ids, query_attention_mask):
        with torch.no_grad():
            doc_input_ids, doc_attention_mask = self.all_docs

            query_encodings, doc_encodings = self.encode(
                query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask
            )

            if self.config.inference_type == "sim":
                query = self.reduce(query_encodings, query_input_ids, self.config.mask_zeros_in_query)
                doc = self.reduce(doc_encodings, doc_input_ids, self.config.mask_zeros_in_doc)

                if self.config.inference_direction == "query":
                    scores = self.tokens_similarity(
                        query,
                        doc,
                        query_attention_mask,
                        doc_attention_mask,
                        with_lens_norm=self.config.length_norm_at_inference,
                    ).T
                elif self.config.inference_direction == "doc":
                    scores = self.tokens_similarity(
                        doc,
                        query,
                        doc_attention_mask,
                        query_attention_mask,
                        with_lens_norm=self.config.length_norm_at_inference,
                    )
                elif self.config.inference_direction == "both":
                    first = self.tokens_similarity(
                        query,
                        doc,
                        query_attention_mask,
                        doc_attention_mask,
                        with_lens_norm=self.config.length_norm_at_inference,
                    ).T
                    second = self.tokens_similarity(
                        doc,
                        query,
                        doc_attention_mask,
                        query_attention_mask,
                        with_lens_norm=self.config.length_norm_at_inference,
                    )
                    scores = (first + second) / 2

            elif self.config.inference_type == "clf":
                scores = self.clf_loss(
                    query_encodings,
                    query_attention_mask,
                )
            elif self.config.inference_type == "pooled_cosine":
                scores = self.pooled_cosine_scores(
                    query_encodings,
                    query_input_ids,
                    query_attention_mask,
                    doc_encodings,
                    doc_input_ids,
                    doc_attention_mask,
                )
            else:
                raise ValueError(f"Unknown inference type: {self.config.inference_type}")

        return scores

    def prepare_inputs(
        self,
        query_input_ids,
        query_attention_mask,
        doc_input_ids,
        doc_attention_mask,
        labels=None,
    ):
        if self.config.num_repeats > 1:
            n = self.config.num_repeats
            query_input_ids = query_input_ids.repeat(n, 1)
            query_attention_mask = query_attention_mask.repeat(n, 1)
            doc_input_ids = doc_input_ids.repeat(n, 1)
            doc_attention_mask = doc_attention_mask.repeat(n, 1)
            labels = labels.repeat(n)

        mlm_labels = None
        if self.config.mask_prob > 0.0:
            query_input_ids, mlm_labels = self.mask_tokens(query_input_ids)

        return (
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
            labels,
            mlm_labels,
        )

    def reduce(self, encodings, input_ids, do_mask_zeros=False):
        projected = self.project(encodings, keep_dims=self.training)

        if do_mask_zeros:
            projected = self.mask_zeros(projected, input_ids)

        return self.normalize(projected)

    def forward(
        self,
        query_input_ids,
        query_attention_mask,
        doc_input_ids=None,
        doc_attention_mask=None,
        labels=None,
    ):
        scores = None
        if not self.training:
            scores = self.inference_forward(query_input_ids, query_attention_mask)

        (
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
            labels,
            mlm_labels,
        ) = self.prepare_inputs(
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
            labels,
        )

        query_encodings, doc_encodings = self.encode(
            query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask
        )

        total_loss = 0.0

        if self.config.sim_factor > 0.0:
            query = self.reduce(query_encodings, query_input_ids, self.config.mask_zeros_in_query)
            doc = self.reduce(doc_encodings, doc_input_ids, self.config.mask_zeros_in_doc)

            sim_loss = self.sim_loss(
                query,
                doc,
                query_attention_mask,
                doc_attention_mask,
                labels,
            )
            total_loss += sim_loss * self.config.sim_factor

        if self.config.mlm_factor > 0.0:
            mlm_loss = self.mlm_loss(query_encodings, mlm_labels)
            total_loss += mlm_loss * self.config.mlm_factor

        if not self.config.pretrain_mode and self.config.requires_classifier_head:
            if self.config.inference_type == "pooled_cosine":
                clf_loss = self.pooled_cosine_loss(
                    query_encodings,
                    query_input_ids,
                    query_attention_mask,
                    labels,
                )
                total_loss += clf_loss * self.config.clf_factor
            else:
                clf_scores = self.clf_loss(query_encodings, query_attention_mask, labels=None)
                if self.config.clf_factor > 0.0:
                    total_loss += self.clf_criterion(clf_scores, labels) * self.config.clf_factor
                if self.config.tversky_loss_weight > 0.0:
                    total_loss += self.tversky_target_loss(clf_scores, labels)
                total_loss += self.lagrangian_recall_loss(clf_scores, labels)

        return SequenceClassifierOutput(
            loss=total_loss,
            logits=scores,
        )

    def sim_loss(
        self,
        query_projections,
        doc_projections,
        query_attention_mask,
        doc_attention_mask,
        labels,
    ):
        sim_mat = None
        lbls = None
        bs = query_projections.size(0) // self.config.num_repeats  # original batch size
        Q, D = query_projections, doc_projections
        if self.config.pretrain_mode:
            if self.config.rep_tokens == "all":
                sim_mat = self.token_sim(Q[:bs, :, :], Q[bs:, :, :], bs)
                a, b = Q[:bs, :1, :], Q[bs:, :1, :]
            elif self.config.rep_tokens == "cls":
                a, b = Q[:bs, :, :], Q[bs:, :, :]

        else:
            # FIX: Use float32 instead of float64 for MPS compatibility
            lbls = Variable(labels.float(), requires_grad=True)
            if self.config.rep_tokens == "all":
                sim_mat = self.query_doc_similarity_matrix(
                    Q,
                    D,
                    query_attention_mask,
                    doc_attention_mask,
                    symetric_mode=self.config.symetric_mode,
                )
                a, b = Q[:, :1, :], D[:, :1, :]
            elif self.config.rep_tokens == "cls":
                a, b = Q, D

        features = torch.cat((a, b), 1)
        sim_loss = self.sim_criterion(features=features, labels=lbls, sim=sim_mat)
        if not self.config.pretrain_mode and self.config.label_anchor_factor > 0.0:
            sim_loss = sim_loss + self.config.label_anchor_factor * self.label_anchor_criterion(features, labels)
        return sim_loss

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def clf_loss(self, encodings, attention_mask, labels=None):
        if self.config.clf_level == "token":
            x = (encodings.last_hidden_state).permute(0, 2, 1)
            logits = (
                (
                    self.clf.weight.view(self.config.head_token_size, self.inner_dim, -1).permute(2, 0, 1)
                    @ x.unsqueeze(1)
                )
                .max(2)
                .values.sum(2)
            )

        elif self.config.clf_level == "cls":
            if "deberta" in self.config.encoder.model_type:
                cls = encodings[0][:, 0, :]
            # FIX: Handle missing pooler_output for models like SecureBERT
            elif hasattr(encodings, "pooler_output") and encodings.pooler_output is not None:
                cls = encodings.pooler_output
            else:
                # Fallback to mean pooling for models without pooler
                cls = self.mean_pooling(encodings, attention_mask)
            logits = self.clf(cls)
        elif self.config.clf_level == "mean_pool":
            cls = self.mean_pooling(encodings, attention_mask)
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            logits = self.clf(cls)
        else:
            raise ValueError(f"Unknown clf level: {self.config.clf_level}")

        clf_scores = logits / self.config.clf_dim

        if labels is None:
            return clf_scores

        return self.clf_criterion(clf_scores, labels)

    def tversky_target_loss(self, logits, labels):
        """Target-class Tversky loss: weight * (1 - Tversky) ** gamma on malicious class."""
        weight = float(self.config.tversky_loss_weight or 0.0)
        label_id = self.config.target_label_id
        if labels is None or label_id is None or weight <= 0.0:
            return logits.new_zeros(())

        probabilities = torch.softmax(logits, dim=1)
        target_probabilities = probabilities[:, int(label_id)]
        target_mask = labels == int(label_id)
        non_target_mask = ~target_mask

        if not target_mask.any():
            return logits.new_zeros(())

        true_positive = target_probabilities[target_mask].sum()
        false_negative = (1.0 - target_probabilities[target_mask]).sum()
        false_positive = target_probabilities[non_target_mask].sum() if non_target_mask.any() else logits.new_zeros(())

        alpha = max(float(self.config.tversky_alpha or 0.0), 0.0)
        beta = max(float(self.config.tversky_beta or 0.0), 0.0)
        gamma = max(float(self.config.tversky_gamma or 1.0), 1e-6)
        denominator = true_positive + alpha * false_positive + beta * false_negative
        tversky_index = true_positive / denominator.clamp_min(1e-6)
        return logits.new_tensor(weight) * torch.pow((1.0 - tversky_index).clamp_min(0.0), gamma)

    def lagrangian_recall_loss(self, logits, labels):
        """Apply the constrained rare-class soft-recall objective.

        The objective combines a recall hinge with optional false-positive and reward
        terms. The recall component is deliberately skipped for batches without a
        target-class example, so sparse malicious batches remain numerically safe.
        """
        recall_weight = self._current_recall_constraint_lambda(logits)
        fp_weight = float(self.config.false_positive_penalty_beta or 0.0)
        reward_weight = float(self.config.constrained_reward_weight or 0.0)
        label_id = self.config.target_label_id
        if (
            labels is None
            or label_id is None
            or (
                not self.config.adaptive_dual_enabled
                and float(recall_weight.detach().item()) <= 0.0
                and fp_weight <= 0.0
                and reward_weight <= 0.0
            )
        ):
            if self.training and labels is not None and label_id is not None and self.config.recall_constraint_rho > 0.0:
                self._would_be_hinge(logits, labels, int(label_id))
            return logits.new_zeros(())

        probabilities = torch.softmax(logits, dim=1)
        rare_probabilities = probabilities[:, int(label_id)]
        loss = logits.new_zeros(())

        if (
            self.training
            and self.config.recall_hinge_window == "accum"
            and not self.config.adaptive_dual_enabled
            and float(recall_weight.detach().item()) > 0.0
            and self.config.recall_constraint_rho > 0.0
        ):
            rare_mask = labels == int(label_id)
            loss = loss + self._window_hinge(rare_probabilities, rare_mask, recall_weight)
        elif (
            self.config.adaptive_dual_enabled or float(recall_weight.detach().item()) > 0.0
        ) and self.config.recall_constraint_rho > 0.0:
            rare_mask = labels == int(label_id)
            if self.training:  # detached activity statistics
                has_mal = bool(rare_mask.any())
                sr = float(rare_probabilities[rare_mask].mean().detach().item()) if has_mal else 0.0
                hv = float(recall_weight.detach().item()) * max(0.0, float(self.config.recall_constraint_rho) - sr)
                hv = hv if has_mal else 0.0
                self._hinge_unit(has_mal, sr, hv)
                self._hinge_micro_window_track(has_mal, hv > 0.0)
            if rare_mask.any():
                soft_recall = rare_probabilities[rare_mask].mean()
                raw_recall_gap = logits.new_tensor(float(self.config.recall_constraint_rho)) - soft_recall
                loss = loss + recall_weight * torch.relu(raw_recall_gap)
                self._update_adaptive_dual(raw_recall_gap)

        if fp_weight > 0.0:
            non_rare_mask = labels != int(label_id)
            if non_rare_mask.any():
                loss = loss + fp_weight * rare_probabilities[non_rare_mask].mean()

        if reward_weight > 0.0:
            reward_metric = self.config.constrained_reward_metric or "none"
            if reward_metric == "soft_accuracy":
                reward = probabilities.gather(1, labels.view(-1, 1)).mean()
                loss = loss - reward_weight * reward
            elif reward_metric == "soft_noise_reduction":
                non_rare_mask = labels != int(label_id)
                if non_rare_mask.any():
                    loss = loss - reward_weight * (1.0 - rare_probabilities[non_rare_mask].mean())
            elif reward_metric == "soft_benign_noise_reduction":
                non_rare_mask = labels != int(label_id)
                benign_label_id = getattr(self.config, "benign_noise_reduction_label_id", None)
                if benign_label_id is not None and non_rare_mask.any():
                    loss = loss - reward_weight * probabilities[:, int(benign_label_id)][non_rare_mask].mean()
            elif reward_metric != "none":
                raise ValueError(f"Unsupported constrained_reward_metric: {reward_metric}")

        return loss

    # ------------------------------------------------------------------ window hinge and hinge-activity statistics
    def hinge_window_begin(self, n_micro):
        """Called by the trainer at the start of every gradient-accumulation window (n_micro micro-batches)."""
        self._hinge_win = {"n": int(n_micro), "seen": 0, "S": 0.0, "M": 0, "buf": None, "any_active": False,
                           "any_mal": False}

    def _hinge_stats(self):
        st = getattr(self, "_hinge_epoch_stats", None)
        if st is None:
            st = self._hinge_epoch_stats = {"units": 0, "units_with_mal": 0, "units_active": 0, "hinge_sum": 0.0,
                                            "softrec_sum": 0.0, "steps": 0, "steps_with_mal": 0, "steps_active": 0}
        return st

    def flush_hinge_epoch(self, epoch):
        """Per-epoch hinge activity (detached statistics; no effect on training). A unit is a micro-batch ('micro')
        or an accumulation window ('accum'); a step is an optimizer step, active if the hinge is > 0 in any unit."""
        st = self._hinge_stats()
        out = {"epoch": epoch, "window": self.config.recall_hinge_window,
               "rho": self.config.recall_constraint_rho, "lambda": self.config.recall_constraint_lambda,
               "beta": self.config.false_positive_penalty_beta, "sim_factor": self.config.sim_factor, **st}
        out["mean_hinge_per_unit"] = st["hinge_sum"] / st["units"] if st["units"] else 0.0
        out["frac_units_active"] = st["units_active"] / st["units"] if st["units"] else 0.0
        out["frac_steps_active"] = st["steps_active"] / st["steps"] if st["steps"] else 0.0
        out["mean_softrec_units_with_mal"] = st["softrec_sum"] / st["units_with_mal"] if st["units_with_mal"] else None
        self._hinge_epoch_stats = None
        return out

    def _hinge_unit(self, has_mal, softrec, hinge_value):
        st = self._hinge_stats()
        st["units"] += 1
        if has_mal:
            st["units_with_mal"] += 1
            st["softrec_sum"] += float(softrec)
        if hinge_value > 0.0:
            st["units_active"] += 1
        st["hinge_sum"] += float(hinge_value)

    def _hinge_micro_window_track(self, has_mal, active):
        """Per-optimizer-step bookkeeping in 'micro' mode (a step is active if any micro-batch hinge is > 0)."""
        win = getattr(self, "_hinge_win", None)
        if win is None:
            return
        win["seen"] += 1
        win["any_mal"] |= bool(has_mal)
        win["any_active"] |= bool(active)
        if win["seen"] == win["n"]:
            st = self._hinge_stats()
            st["steps"] += 1
            st["steps_with_mal"] += int(win["any_mal"])
            st["steps_active"] += int(win["any_active"])
            self._hinge_win = None

    def _window_hinge(self, rare_probabilities, rare_mask, recall_weight):
        """Soft-recall hinge over the gradient-accumulation window (recall_hinge_window='accum').

        Window loss H = lambda * relu(rho - sum_i p_i / M) over all M malicious alerts of the window. Micro-batch
        graphs are freed after each backward, so for every micro-batch the gradient of s_k = sum of its malicious
        p_i is stored in a buffer (torch.autograd.grad, retain_graph) and the final micro-batch returns
        n_micro * [H with the gradient routed to all micro-batches]. The trainer divides every micro-batch loss by
        the accumulation steps, so the step objective is mean_k(CE_k + SupCon_k) + H; with n_micro = 1 this is the
        per-micro-batch hinge exactly. A window without a malicious alert contributes 0.
        """
        win = getattr(self, "_hinge_win", None)
        assert win is not None, "the 'accum' hinge needs hinge_window_begin() from the trainer"
        rho = float(self.config.recall_constraint_rho)
        lam = float(recall_weight.detach().item())
        params = getattr(self, "_hinge_params", None)
        if params is None:
            params = self._hinge_params = [p for p in self.parameters() if p.requires_grad]
        win["seen"] += 1
        last = win["seen"] == win["n"]
        m_k = int(rare_mask.sum().item())
        s_k = rare_probabilities[rare_mask].sum() if m_k else None
        loss = rare_probabilities.new_zeros(())
        if m_k and not last and torch.is_grad_enabled() and lam > 0.0:
            grads = torch.autograd.grad(s_k, params, retain_graph=True, allow_unused=True)
            if win["buf"] is None:
                win["buf"] = [None] * len(params)
            for j, g in enumerate(grads):
                if g is not None:
                    win["buf"][j] = g.detach() if win["buf"][j] is None else win["buf"][j] + g.detach()
        if m_k:
            win["S"] += float(s_k.detach().item())
            win["M"] += m_k
        if not last:
            return loss
        st = self._hinge_stats()
        st["steps"] += 1
        M = win["M"]
        if M == 0:  # skipped only when the whole window has no malicious alert
            self._hinge_unit(False, 0.0, 0.0)
            self._hinge_win = None
            return loss
        st["steps_with_mal"] += 1
        R = win["S"] / M
        h_val = lam * max(0.0, rho - R)
        self._hinge_unit(True, R, h_val)
        if h_val > 0.0:
            st["steps_active"] += 1
            coef = -lam / M
            term = rare_probabilities.new_zeros(())
            if s_k is not None:
                term = term + coef * s_k
            if win["buf"] is not None:
                for p, b in zip(params, win["buf"]):
                    if b is not None:
                        term = term + coef * (b * p).sum()
            # value = H, gradient = dH/dtheta over the whole window; scaled by n_micro (see docstring)
            loss = loss + win["n"] * (term - term.detach() + h_val)
        self._hinge_win = None
        return loss

    @torch.no_grad()
    def _would_be_hinge(self, logits, labels, label_id):
        """lambda=0 (CE) runs: the soft recall the hinge would see, at lambda=1 (detached; no effect on training)."""
        p = torch.softmax(logits.detach().float(), dim=1)[:, label_id]
        mask = labels == label_id
        rho = float(self.config.recall_constraint_rho)
        if self.config.recall_hinge_window == "micro":
            has = bool(mask.any())
            sr = float(p[mask].mean().item()) if has else 0.0
            hv = max(0.0, rho - sr) if has else 0.0
            self._hinge_unit(has, sr, hv)
            self._hinge_micro_window_track(has, hv > 0.0)
            return
        win = getattr(self, "_hinge_win", None)
        if win is None:
            return
        win["seen"] += 1
        win["S"] += float(p[mask].sum().item())
        win["M"] += int(mask.sum().item())
        if win["seen"] == win["n"]:
            st = self._hinge_stats()
            st["steps"] += 1
            if win["M"]:
                R = win["S"] / win["M"]
                hv = max(0.0, rho - R)
                st["steps_with_mal"] += 1
                st["steps_active"] += int(hv > 0.0)
                self._hinge_unit(True, R, hv)
            else:
                self._hinge_unit(False, 0.0, 0.0)
            self._hinge_win = None

    def _current_recall_constraint_lambda(self, logits):
        if self.config.adaptive_dual_enabled:
            return self._adaptive_dual_lambda.detach().clone().to(device=logits.device, dtype=logits.dtype)
        return logits.new_tensor(float(self.config.recall_constraint_lambda or 0.0))

    def _update_adaptive_dual(self, recall_gap):
        if not (self.training and self.config.adaptive_dual_enabled):
            return

        with torch.no_grad():
            self._adaptive_dual_steps.add_(1)
            steps = int(self._adaptive_dual_steps.item())
            if steps <= int(self.config.adaptive_dual_warmup_steps):
                return
            interval = max(1, int(self.config.adaptive_dual_update_interval_steps))
            if (steps - int(self.config.adaptive_dual_warmup_steps)) % interval != 0:
                return

            decay = min(max(float(self.config.adaptive_dual_ema_decay or 0.0), 0.0), 0.999)
            detached_gap = recall_gap.detach().to(device=self._adaptive_dual_gap_ema.device, dtype=torch.float32)
            if steps == int(self.config.adaptive_dual_warmup_steps) + 1:
                self._adaptive_dual_gap_ema.copy_(detached_gap)
            else:
                self._adaptive_dual_gap_ema.mul_(decay).add_(detached_gap, alpha=1.0 - decay)

            eta = float(self.config.adaptive_dual_eta or 0.0)
            max_lambda = max(0.0, float(self.config.adaptive_dual_max_lambda or 0.0))
            updated_lambda = torch.clamp(
                self._adaptive_dual_lambda + eta * self._adaptive_dual_gap_ema,
                min=0.0,
                max=max_lambda,
            )
            self._adaptive_dual_lambda.copy_(updated_lambda)

    def pooled_embedding(self, encodings, input_ids, attention_mask, do_mask_zeros=False):
        projected = self.project(encodings, keep_dims=self.training)
        mask = attention_mask.type(projected.dtype).to(projected.device)
        if do_mask_zeros:
            projected = self.mask_zeros(projected, input_ids)
            mask = mask * (input_ids != 0).type(projected.dtype).to(projected.device)

        pooled = (projected * mask.unsqueeze(2)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)
        return F.normalize(pooled, p=2, dim=1)

    def pooled_cosine_scale(self):
        if self.config.pooled_cosine_learn_scale:
            return self.pooled_cosine_log_scale.exp().clamp(max=100.0)
        return next(self.parameters()).new_tensor(float(self.config.pooled_cosine_scale))

    def pooled_cosine_scores(
        self,
        query_encodings,
        query_input_ids,
        query_attention_mask,
        doc_encodings,
        doc_input_ids,
        doc_attention_mask,
    ):
        query = self.pooled_embedding(
            query_encodings,
            query_input_ids,
            query_attention_mask,
            self.config.mask_zeros_in_query,
        )
        doc = self.pooled_embedding(
            doc_encodings,
            doc_input_ids,
            doc_attention_mask,
            self.config.mask_zeros_in_doc,
        )
        return self.pooled_cosine_scale() * (query @ doc.T)

    def pooled_cosine_loss(self, query_encodings, query_input_ids, query_attention_mask, labels):
        doc_input_ids, doc_attention_mask = self.all_docs
        doc_input_ids = doc_input_ids.to(query_input_ids.device)
        doc_attention_mask = doc_attention_mask.to(query_attention_mask.device)
        doc_encodings = self.encode_one(doc_input_ids, doc_attention_mask)
        scores = self.pooled_cosine_scores(
            query_encodings,
            query_input_ids,
            query_attention_mask,
            doc_encodings,
            doc_input_ids,
            doc_attention_mask,
        )
        return self.clf_criterion(scores, labels)

    def mlm_loss(self, encodings, mlm_lables):
        sequence_output = encodings[0]
        prediction_scores = self.lm_head(sequence_output)
        vocab_size = (
            self.tokenizer.vocab_size
            if self.encoder.config.model_type != "deberta-v2"
            else self.tokenizer.vocab_size + 100
        )
        mlm_loss = self.mlm_criterion(prediction_scores.view(-1, vocab_size), mlm_lables.view(-1))
        return mlm_loss

    def encode(
        self,
        query_input_ids,
        query_attention_mask,
        doc_input_ids,
        doc_attention_mask,
        do_zero_mask=False,
    ):
        if self.config.encoding_type == "dual":
            query_encodings = self.encode_one(query_input_ids, query_attention_mask)
            doc_input_ids, doc_attention_mask = (
                doc_input_ids.to(query_input_ids.device),
                doc_attention_mask.to(query_attention_mask.device),
            )
            doc_encodings = self.encode_one(doc_input_ids, doc_attention_mask)

        elif self.config.encoding_type == "cross":
            raise NotImplementedError("Cross encoder not implemented yet")
        else:
            raise ValueError(f"Unknown encoding type: {self.config.encoding_type}")

        if do_zero_mask:
            query_encodings[0][query_attention_mask] = 0.0
            doc_encodings[0][doc_attention_mask] = 0.0

        return query_encodings, doc_encodings

    def encode_one(self, input_ids, attention_mask):
        # T4 (SM75) memory-efficient SDPA produces NaN for padded sequences >~50 tokens.
        # Force the numerically-stable math backend on CUDA for both training and inference.
        ctx = _math_sdpa_ctx() if torch.cuda.is_available() else nullcontext()
        with ctx:
            encodings = self.encoder(input_ids, attention_mask=attention_mask)
        return encodings

    def project(self, encodings, keep_dims=True):
        if self.config.rep_tokens == "all":
            encoded = encodings[0]
            if keep_dims:
                encoded = self.dropout(encoded)
        elif self.config.rep_tokens == "cls":
            cls = encodings.last_hidden_state[:, 0, :]
            if keep_dims:
                cls = self.batch_norm(cls)
                cls = self.dropout(cls)
            encoded = cls.unsqueeze(1)

        projected = self.projection(encoded)
        return projected

    def normalize(self, tensor):
        return torch.nn.functional.normalize(tensor, p=2, dim=2)

    def mask_zeros(self, tensor, input_ids):
        return tensor * (input_ids != 0).float().unsqueeze(2)

    def query_doc_similarity_matrix(self, Q, D, Q_mask, D_mask, symetric_mode=True):
        QD = self.tokens_similarity(D, Q, D_mask, Q_mask)
        QQ = self.tokens_similarity(Q, Q, Q_mask, Q_mask)
        DD = self.tokens_similarity(D, D, D_mask, D_mask)

        if symetric_mode:
            DQ = QD.T
        else:
            DQ = self.tokens_similarity(Q, D, Q_mask, D_mask)

        return torch.cat((torch.cat((QQ, DQ)), torch.cat((QD, DD))), 1)

    def tokens_similarity(self, B1, B2, B1_mask=None, B2_mask=None, with_lens_norm=True):
        tokens_sim = B1 @ B2.permute(0, 2, 1).unsqueeze(1)
        lens = B2.shape[1]

        if self.config.length_norm:
            if B1_mask is None or B2_mask is None:
                raise ValueError("tokens_similarity has to have values for B1_mask and B2_mask when length_norm=True")
            B1_mask = B1_mask.type(B1.dtype).to(B1.device)
            B2_mask = B2_mask.type(B2.dtype).to(B2.device)
            tokens_mask = B1_mask.unsqueeze(-1) @ B2_mask.unsqueeze(-1).permute(0, 2, 1).unsqueeze(
                1
            )  # [n_d, n_q, l_d, l_q]
            lens = tokens_mask.max(2)[0].sum(2)
            tokens_sim = tokens_sim * tokens_mask

        scores = tokens_sim.max(2)[0].sum(2)

        if with_lens_norm:
            scores = scores / lens

        return scores

    def mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, self.config.mlm_prob)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[torch.where(inputs == 0)] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        indices_random = indices_random.to(inputs.device)
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long).to(inputs.device)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class FastFit(FastFitTrainable):
    def forward(self, input_ids, attention_mask, labels=None):
        return SequenceClassifierOutput(
            logits=self.inference_forward(input_ids, attention_mask),
        )


# main tests:

if __name__ == "__main__":
    model = FastFitTrainable.from_pretrined_encoder("roberta-base")
