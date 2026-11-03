"""Text baselines used on the production tenants: frozen-SecureBERT kNN, SetFit-LoRA and Qwen3-0.6B-LoRA.

They consume the same rendered alert text as the encoder models. None of them was run on GUIDE in the paper,
but they run on any split produced by ``scripts/build_guide_splits.py`` via ``scripts/train_guide.py``.
"""

from __future__ import annotations

import tempfile
from typing import Any

import numpy as np


class KNNBaseline:
    """k-nearest neighbours (k=5, cosine) over frozen sentence embeddings; a stand-in for a verdict cache.

    Class scores are summed neighbour similarities 1/(1+distance), normalized to sum to one."""

    def __init__(self, recipe: dict[str, Any]):
        from sentence_transformers import SentenceTransformer
        from sklearn.neighbors import KNeighborsClassifier

        self.embedder = SentenceTransformer(recipe.get("embedding_model", "cisco-ai/SecureBERT2.0-biencoder"))
        self.knn = KNeighborsClassifier(n_neighbors=recipe.get("k_neighbors", 5), metric="cosine")
        self.classes_ = None

    def fit(self, texts, labels, val_texts=None, val_labels=None):
        self.classes_ = np.unique(labels)
        self.train_labels_ = np.asarray(labels)
        self.knn.fit(self.embedder.encode(texts, convert_to_numpy=True), labels)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        emb = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        dist, idx = self.knn.kneighbors(emb)
        sims = 1.0 / (1.0 + dist)
        cidx = {c: i for i, c in enumerate(self.classes_)}
        proba = np.zeros((len(emb), len(self.classes_)))
        for i, nb in enumerate(idx):
            for lab, s in zip(self.train_labels_[nb], sims[i]):
                proba[i, cidx[lab]] += s
        zero = proba.sum(axis=1) == 0
        if zero.any():
            for r, pred in zip(np.flatnonzero(zero), self.knn.predict(emb[zero])):
                proba[r, cidx[pred]] = 1.0
        zero = proba.sum(axis=1) == 0
        proba[zero, :] = 1.0 / len(self.classes_)
        return proba / proba.sum(axis=1, keepdims=True)


class SetFitLoRA:
    """SetFit (paraphrase-mpnet-base-v2) with a LoRA adapter (r=16, alpha=32) on q/k/v/o projections."""

    def __init__(self, recipe: dict[str, Any]):
        self.r = recipe
        self.model = None
        self.classes_ = None

    def fit(self, texts, labels, val_texts=None, val_labels=None):
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from setfit import SetFitModel, Trainer, TrainingArguments

        r = self.r
        self.classes_ = np.unique(labels)
        self.model = SetFitModel.from_pretrained(r.get("model_name", "sentence-transformers/paraphrase-mpnet-base-v2"),
                                                 labels=self.classes_.tolist())
        if r.get("use_lora", False):
            self.model.model_body.add_adapter(LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, r=int(r.get("lora_r", 16)), lora_alpha=int(r.get("lora_alpha", 32)),
                lora_dropout=float(r.get("lora_dropout", 0.1)),
                target_modules=r.get("lora_target_modules", ["q", "k", "v", "o"]), bias="none"))
        ev = Dataset.from_dict({"text": val_texts, "label": val_labels}) if val_texts else None
        args = TrainingArguments(batch_size=int(r.get("batch_size", 16)), num_epochs=int(r.get("num_train_epochs", 5)),
                                 num_iterations=int(r.get("num_iterations", 20)),
                                 body_learning_rate=float(r.get("learning_rate", 2e-5)), seed=int(r.get("seed", 42)),
                                 report_to="none")
        Trainer(model=self.model, args=args, train_dataset=Dataset.from_dict({"text": texts, "label": labels}),
                eval_dataset=ev, column_mapping={"text": "text", "label": "label"}).train()
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return np.asarray(self.model.predict_proba(texts))


class QwenLoRA:
    """Qwen3-0.6B-Base sequence classifier with LoRA (r=16, alpha=32, q/k/v/o) and a trained score head."""

    def __init__(self, recipe: dict[str, Any]):
        self.r = recipe
        self.classes_ = None

    def fit(self, texts, labels, val_texts=None, val_labels=None):
        import torch
        from datasets import Dataset, DatasetDict
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

        r = self.r
        fp16 = bool(r.get("fp16", True))
        self.max_len = int(r.get("max_seq_length", 2048))
        self.eval_bs = int(r.get("per_device_eval_batch_size", 8))
        self.classes_ = np.unique(labels)
        l2i = {lab: i for i, lab in enumerate(self.classes_)}
        name = r.get("model_name", "Qwen/Qwen3-0.6B-Base")
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=len(self.classes_), torch_dtype=torch.float16 if fp16 else torch.float32)
        model.config.pad_token_id = self.tok.pad_token_id
        if r.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()

        def prep(ex):
            t = self.tok(ex["text"], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
            t["labels"] = [l2i[x] for x in ex["label"]]
            return t

        dd = {"train": Dataset.from_dict({"text": texts, "label": labels})}
        if val_texts is not None:
            dd["validation"] = Dataset.from_dict({"text": val_texts, "label": val_labels})
        dd = DatasetDict({k: v.map(prep, batched=True, remove_columns=["text", "label"]) for k, v in dd.items()})
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=int(r.get("lora_r", 16)), lora_alpha=int(r.get("lora_alpha", 32)),
            lora_dropout=float(r.get("lora_dropout", 0.1)), target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=["score"], inference_mode=False))
        args = TrainingArguments(
            output_dir=tempfile.mkdtemp(prefix="qwen_lora_"), num_train_epochs=r.get("num_train_epochs", 5),
            per_device_train_batch_size=r.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=self.eval_bs,
            gradient_accumulation_steps=r.get("gradient_accumulation_steps", 8),
            learning_rate=float(r.get("learning_rate", 2e-4)), fp16=fp16, save_strategy="no",
            eval_strategy="no" if val_texts is None else "epoch", logging_steps=10, report_to="none",
            dataloader_pin_memory=False, remove_unused_columns=True)
        Trainer(model=model, args=args, train_dataset=dd["train"], eval_dataset=dd.get("validation")).train()
        self.model = model
        return self

    def predict_proba(self, texts) -> np.ndarray:
        import torch

        self.model.eval()
        dev = next(self.model.parameters()).device
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.eval_bs):
                t = self.tok(texts[i:i + self.eval_bs], padding=True, truncation=True, max_length=self.max_len,
                             return_tensors="pt")
                out.append(self.model(input_ids=t["input_ids"].to(dev), attention_mask=t["attention_mask"].to(dev)).logits.cpu())
        return torch.softmax(torch.cat(out), dim=-1).numpy()
