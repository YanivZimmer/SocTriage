"""SecureBERT 2.0 + LoRA classifier trained with the FastFit trainer (RiskFit, CE, focal-Tversky, MaxSim).

One class covers every encoder model in the paper; the YAML recipe selects the objective:

* RiskFit:        clf_loss_factor=1, sim_loss_factor=0.5 (SupCon), recall_constraint_rho/lambda > 0
                  (hinge on soft malicious recall), optional false_positive_penalty_beta (beta)
* CE:             same as RiskFit with recall_constraint_lambda=0 (or the hinge keys omitted)
* focal-Tversky:  CE recipe plus tversky_loss_weight/alpha/beta/gamma
* MaxSim:         clf_loss_factor=0, sim_loss_factor=1, inference_type=sim (FastFit similarity classifier)

All share the backbone, LoRA (rank 32, alpha 64, dropout 0.1 on Wqkv/Wo), Adafactor, micro-batches of 4
sampled 50/50 malicious/non-malicious with 9 gradient-accumulation steps, and early stopping on validation
accuracy. ``FastFitLoRAConfig.from_dict`` maps a recipe YAML to the config exactly as the code that produced
the paper's runs did (keys it does not list keep their defaults).
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

MALICIOUS = "True Positive - Malicious"


# --------------------------------------------------------------------------- trainer helpers


def build_early_stopping_extras(
    *, has_validation: bool, early_stopping_patience: int, metric_for_best_model: str, greater_is_better: bool
) -> tuple[dict[str, Any], list[Any]]:
    """HF TrainingArguments updates and callbacks for per-epoch validation early stopping + best reload."""
    if not has_validation or early_stopping_patience <= 0:
        return {}, []
    from transformers import EarlyStoppingCallback

    return (
        {
            "do_eval": True,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": metric_for_best_model,
            "greater_is_better": greater_is_better,
        },
        [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )


class FocalLoss(torch.nn.Module):
    """FL(p_t) = -(1 - p_t)^gamma * log(p_t), optionally class-weighted."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma, self.weight, self.reduction = gamma, weight, reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        loss = ((1 - torch.exp(-ce)) ** self.gamma) * ce
        return loss if self.reduction == "none" else loss.mean()


class AsymmetricMaliciousLoss(torch.nn.Module):
    """Extra loss on true-malicious samples predicted non-malicious (off in every paper recipe)."""

    def __init__(self, *, malicious_label_id: int | None, false_negative_multiplier: float = 1.0,
                 weight: torch.Tensor | None = None, gamma: float = 0.0):
        super().__init__()
        self.malicious_label_id = malicious_label_id
        self.false_negative_multiplier = false_negative_multiplier
        self.weight, self.gamma = weight, gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.gamma > 0.0:
            base = FocalLoss(gamma=self.gamma, weight=self.weight, reduction="none")(logits, labels)
        else:
            base = torch.nn.functional.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        if self.false_negative_multiplier != 1.0 and self.malicious_label_id is not None:
            fn = (labels == self.malicious_label_id) & (logits.argmax(dim=1) != self.malicious_label_id)
            base = torch.where(fn, base * self.false_negative_multiplier, base)
        return base.mean()


def resolve_label_classes(train_labels: list[str], val_labels: list[str] | None = None) -> np.ndarray:
    """Sorted union of train and validation labels (the class order of the FastFit head)."""
    return np.unique(list(train_labels) + list(val_labels or []))


def build_class_weights(labels, label_to_id, class_weight_mode, malicious_weight_multiplier=1.0):
    if class_weight_mode in (None, "none"):
        if malicious_weight_multiplier == 1.0:
            return None
        weights = torch.ones(len(label_to_id), dtype=torch.float32)
    elif class_weight_mode == "balanced":
        ids = np.array([label_to_id[label] for label in labels], dtype=np.int64)
        counts = np.bincount(ids, minlength=len(label_to_id))
        weights = torch.tensor(counts.sum() / (len(label_to_id) * np.maximum(counts, 1)), dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported class_weight_mode: {class_weight_mode}")
    if malicious_weight_multiplier != 1.0 and MALICIOUS in label_to_id:
        weights[label_to_id[MALICIOUS]] *= malicious_weight_multiplier
    return weights


def apply_malicious_inference_delta(probs, label_to_id, malicious_label_name, delta, inference_type):
    """Optional inference-time bias toward the malicious class (delta=0 in every paper recipe)."""
    if inference_type != "clf" or delta == 0.0:
        return probs
    if delta < 0.0:
        raise ValueError("malicious_inference_delta must be non-negative")
    adjusted = probs.clone()
    adjusted[:, label_to_id[malicious_label_name]] += delta
    return adjusted / adjusted.sum(dim=1, keepdim=True).clamp_min(1e-12)


# --------------------------------------------------------------------------- config


@dataclass
class FastFitLoRAConfig:
    model_name: str = "cisco-ai/SecureBERT2.0-biencoder"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(default_factory=lambda: ["Wqkv", "Wo"])
    num_train_epochs: int = 25
    per_device_train_batch_size: int = 6
    per_device_eval_batch_size: int = 12
    max_text_length: int = 1024
    max_label_length: int = 32
    num_repeats: int = 1
    clf_loss_factor: float = 0.0
    sim_loss_factor: float = 1.0
    inference_type: str = "sim"
    clf_level: str = "cls"
    tversky_loss_weight: float = 0.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    tversky_gamma: float = 1.0
    recall_constraint_rho: float = 0.0
    recall_constraint_lambda: float = 0.0
    false_positive_penalty_beta: float = 0.0
    constrained_reward_metric: str = "none"
    constrained_reward_weight: float = 0.0
    adaptive_dual_enabled: bool = False
    adaptive_dual_initial_lambda: float = 0.0
    adaptive_dual_eta: float = 0.05
    adaptive_dual_max_lambda: float = 5.0
    adaptive_dual_ema_decay: float = 0.9
    adaptive_dual_warmup_steps: int = 0
    adaptive_dual_update_interval_steps: int = 1
    pooled_cosine_scale: float = 1.0
    pooled_cosine_learn_scale: bool = False
    malicious_inference_delta: float = 0.0
    length_norm_at_inference: bool = True
    learning_rate: float = 1.0e-4
    warmup_ratio: float = 0.1
    fp16: bool = False
    optimizer: str = "adafactor"
    gradient_accumulation_steps: int = 6
    max_grad_norm: float = 1.0
    output_dir: str | None = None
    early_stopping_patience: int = 0
    metric_for_best_model: str = "eval_accuracy"
    greater_is_better: bool = True
    class_weight_mode: str | None = "balanced"
    focal_loss_gamma: float = 0.0
    malicious_weight_multiplier: float = 1.0
    malicious_false_negative_multiplier: float = 1.0
    scores_temp: float = 0.07
    contrastive_variant: str = "class_balanced"
    contrastive_label_mode: str = "label"
    label_anchor_factor: float = 0.5
    target_positive_margin: float = 0.15
    target_anchor_weight: float = 0.25
    target_hard_negative_strength: float = 0.0
    target_hard_positive_strength: float = 0.0
    target_logit_scale: float = 1.0
    balance_malicious_batches: bool = True
    malicious_label_name: str = MALICIOUS
    logit_norm: bool = False
    seed: int = 42  # LoRA init, dropout, data order and the balanced batch sampler

    # Recipe keys honoured by from_dict (the others keep their defaults, as in the original runs).
    _RECIPE_KEYS = (
        "use_lora", "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules", "num_train_epochs",
        "per_device_train_batch_size", "per_device_eval_batch_size", "max_text_length", "num_repeats",
        "clf_loss_factor", "sim_loss_factor", "inference_type", "clf_level", "tversky_loss_weight",
        "tversky_alpha", "tversky_beta", "tversky_gamma", "recall_constraint_rho", "recall_constraint_lambda",
        "false_positive_penalty_beta", "pooled_cosine_scale", "pooled_cosine_learn_scale",
        "malicious_inference_delta", "length_norm_at_inference", "learning_rate", "warmup_ratio", "fp16",
        "optimizer", "gradient_accumulation_steps", "max_grad_norm", "early_stopping_patience",
        "metric_for_best_model", "greater_is_better", "class_weight_mode", "focal_loss_gamma",
        "malicious_weight_multiplier", "malicious_false_negative_multiplier", "scores_temp",
        "contrastive_variant", "label_anchor_factor", "target_positive_margin", "target_anchor_weight",
        "target_hard_negative_strength", "target_hard_positive_strength", "target_logit_scale",
        "balance_malicious_batches", "malicious_label_name", "seed",
    )

    @classmethod
    def from_dict(cls, recipe: dict[str, Any]) -> "FastFitLoRAConfig":
        kw = {k: recipe[k] for k in cls._RECIPE_KEYS if k in recipe}
        if "model_name_or_path" in recipe:
            kw["model_name"] = recipe["model_name_or_path"]
        if "learning_rate" in kw:
            kw["learning_rate"] = float(kw["learning_rate"])
        if "seed" in kw:
            kw["seed"] = int(kw["seed"])
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in kw.items() if k in names})


# --------------------------------------------------------------------------- classifier


class FastFitLoRAClassifier:
    def __init__(self, config: FastFitLoRAConfig):
        self.config = config
        self.classes_ = None
        self.label_to_id = None
        self.id_to_label = None
        self._trainer = self._model = self._tokenizer = None

    def _apply_lora(self, model) -> None:
        if not self.config.use_lora:
            return
        from peft import LoraConfig, get_peft_model

        model.encoder = get_peft_model(
            model.encoder,
            LoraConfig(r=self.config.lora_r, lora_alpha=self.config.lora_alpha, lora_dropout=self.config.lora_dropout,
                       target_modules=self.config.lora_target_modules, bias="none"),
        )

    def fit(self, texts: list[str], labels: list[str], val_texts: list[str] | None = None,
            val_labels: list[str] | None = None) -> "FastFitLoRAClassifier":
        from datasets import Dataset, DatasetDict

        from .fastfit.train import FastFitTrainer

        if not texts or len(texts) != len(labels):
            raise ValueError("need a non-empty training set with one label per text")
        c = self.config
        self.classes_ = resolve_label_classes(labels, val_labels)
        self.label_to_id = {label: i for i, label in enumerate(self.classes_)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        class_weights = build_class_weights(labels, self.label_to_id, c.class_weight_mode, c.malicious_weight_multiplier)

        dd = {"train": Dataset.from_dict({"text": texts, "label": labels})}
        has_val = bool(val_texts) and bool(val_labels)
        if has_val:
            dd["validation"] = Dataset.from_dict({"text": val_texts, "label": val_labels})
        dd["test"] = Dataset.from_dict({"text": [], "label": []})  # FastFit needs a test split for task "custom"
        dd = DatasetDict(dd)

        output_dir = c.output_dir or tempfile.mkdtemp(prefix="fastfitlora_")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        es_updates, callbacks = build_early_stopping_extras(
            has_validation=has_val, early_stopping_patience=c.early_stopping_patience,
            metric_for_best_model=c.metric_for_best_model, greater_is_better=c.greater_is_better)
        training_args = {
            "output_dir": output_dir, "overwrite_output_dir": True, "do_train": True, "do_eval": False,
            "do_predict": False, "num_train_epochs": c.num_train_epochs,
            "per_device_train_batch_size": c.per_device_train_batch_size,
            "per_device_eval_batch_size": c.per_device_eval_batch_size,
            "gradient_accumulation_steps": c.gradient_accumulation_steps, "learning_rate": c.learning_rate,
            "warmup_ratio": c.warmup_ratio, "fp16": c.fp16, "optim": c.optimizer, "max_grad_norm": c.max_grad_norm,
            "save_strategy": "no", "eval_strategy": "no", "logging_steps": 10, "report_to": "none",
            "dataloader_pin_memory": False, "seed": int(c.seed), **es_updates,
        }
        data_args = {"text_column_name": "text", "label_column_name": "label", "max_text_length": c.max_text_length,
                     "max_label_length": c.max_label_length, "task_name": "custom", "metric_name": "accuracy"}
        config_args = {
            "num_repeats": c.num_repeats, "clf_loss_factor": c.clf_loss_factor, "sim_factor": c.sim_loss_factor,
            "inference_type": c.inference_type, "clf_level": c.clf_level,
            "tversky_loss_weight": c.tversky_loss_weight, "tversky_alpha": c.tversky_alpha,
            "tversky_beta": c.tversky_beta, "tversky_gamma": c.tversky_gamma,
            "recall_constraint_rho": c.recall_constraint_rho, "recall_constraint_lambda": c.recall_constraint_lambda,
            "false_positive_penalty_beta": c.false_positive_penalty_beta,
            "constrained_reward_metric": c.constrained_reward_metric,
            "constrained_reward_weight": c.constrained_reward_weight,
            "adaptive_dual_enabled": c.adaptive_dual_enabled,
            "adaptive_dual_initial_lambda": c.adaptive_dual_initial_lambda, "adaptive_dual_eta": c.adaptive_dual_eta,
            "adaptive_dual_max_lambda": c.adaptive_dual_max_lambda,
            "adaptive_dual_ema_decay": c.adaptive_dual_ema_decay,
            "adaptive_dual_warmup_steps": c.adaptive_dual_warmup_steps,
            "adaptive_dual_update_interval_steps": c.adaptive_dual_update_interval_steps,
            "pooled_cosine_scale": c.pooled_cosine_scale, "pooled_cosine_learn_scale": c.pooled_cosine_learn_scale,
            "scores_temp": c.scores_temp, "contrastive_variant": c.contrastive_variant,
            "contrastive_label_mode": c.contrastive_label_mode, "label_anchor_factor": c.label_anchor_factor,
            "target_label_id": self.label_to_id.get(c.malicious_label_name),
            "target_positive_margin": c.target_positive_margin, "target_anchor_weight": c.target_anchor_weight,
            "target_hard_negative_strength": c.target_hard_negative_strength,
            "target_hard_positive_strength": c.target_hard_positive_strength,
            "target_logit_scale": c.target_logit_scale, "length_norm_at_inference": c.length_norm_at_inference,
            "logit_norm": c.logit_norm,
        }
        self._trainer = FastFitTrainer(
            train_dataset=dd["train"], validation_dataset=dd.get("validation"), test_dataset=dd["test"],
            is_command_line_mode=False, trainer_callbacks=callbacks,
            balance_malicious_batches=c.balance_malicious_batches, malicious_label_name=c.malicious_label_name,
            **training_args, model_name_or_path=c.model_name, **data_args, **config_args,
        )
        mal_id = self.label_to_id.get(c.malicious_label_name)
        if class_weights is not None or c.focal_loss_gamma > 0.0:
            dev = next(self._trainer.model.parameters()).device
            w = class_weights.to(dev) if class_weights is not None else None
            if c.focal_loss_gamma > 0.0 or c.malicious_false_negative_multiplier != 1.0:
                self._trainer.model.clf_criterion = AsymmetricMaliciousLoss(
                    gamma=c.focal_loss_gamma, weight=w, malicious_label_id=mal_id,
                    false_negative_multiplier=c.malicious_false_negative_multiplier)
            else:
                self._trainer.model.clf_criterion = torch.nn.CrossEntropyLoss(weight=w)
        elif c.malicious_false_negative_multiplier != 1.0:
            self._trainer.model.clf_criterion = AsymmetricMaliciousLoss(
                malicious_label_id=mal_id, false_negative_multiplier=c.malicious_false_negative_multiplier)

        self._apply_lora(self._trainer.model)  # LoRA is applied after the trainer builds the model, as in the runs
        self._trainer.train()
        self._model, self._tokenizer = self._trainer.model, self._trainer.tokenizer
        return self

    def _logits(self, texts: list[str]) -> torch.Tensor:
        bs = self.config.per_device_eval_batch_size
        if not texts:
            return torch.empty((0, len(self.classes_)), dtype=torch.float32)
        self._model.eval()
        device = next(self._model.parameters()).device
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                tok = self._tokenizer(texts[i:i + bs], padding=True, truncation=True,
                                      max_length=self.config.max_text_length, return_tensors="pt",
                                      return_attention_mask=True)
                logits = self._model.inference_forward(tok["input_ids"].to(device), tok["attention_mask"].to(device))
                if torch.isnan(logits).any():
                    logger.error("NaN in inference logits; affected rows fall back to uniform probabilities")
                    logits = torch.nan_to_num(logits, nan=0.0)
                if self.config.logit_norm:
                    z = logits - logits.mean(dim=1, keepdim=True)
                    logits = z / z.norm(dim=1, keepdim=True).clamp(min=1e-8)
                out.append(logits.cpu())
        return torch.cat(out, dim=0)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        probs = torch.softmax(self._logits(texts), dim=1)
        c = self.config
        probs = apply_malicious_inference_delta(probs, self.label_to_id, c.malicious_label_name,
                                                c.malicious_inference_delta, c.inference_type)
        return probs.numpy()

    def predict(self, texts: list[str]) -> np.ndarray:
        return np.array([self.id_to_label[i] for i in self.predict_proba(texts).argmax(axis=1)])
