"""Model factory: ``build_model(recipe)`` returns an object with fit / predict_proba / classes_.

``recipe["model_type"]`` selects the family (keys as in configs/models/*.yaml):
  fastfitlora -> RiskFit, CE, focal-Tversky, MaxSim     (input: rendered text)
  knn         -> frozen SecureBERT kNN                   (input: rendered text)
  setfit      -> SetFit-LoRA                             (input: rendered text)
  qwen_lora   -> Qwen3-0.6B-LoRA                         (input: rendered text)
  lightgbm    -> LightGBM on structured fields           (input: alert records)
"""

from __future__ import annotations

from typing import Any

TEXT_INPUT = {"fastfitlora", "knn", "setfit", "qwen_lora"}


def build_model(recipe: dict[str, Any]):
    kind = recipe["model_type"]
    if kind == "fastfitlora":
        from .fastfit_lora import FastFitLoRAClassifier, FastFitLoRAConfig

        return FastFitLoRAClassifier(FastFitLoRAConfig.from_dict(recipe))
    if kind == "lightgbm":
        from .tabular import LightGBMModel

        return LightGBMModel(recipe)
    if kind == "knn":
        from .baselines import KNNBaseline

        return KNNBaseline(recipe)
    if kind == "setfit":
        from .baselines import SetFitLoRA

        return SetFitLoRA(recipe)
    if kind == "qwen_lora":
        from .baselines import QwenLoRA

        return QwenLoRA(recipe)
    raise ValueError(f"unknown model_type {kind!r}")
