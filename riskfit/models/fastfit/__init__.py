"""Vendored FastFit (IBM, Apache-2.0) with the RiskFit objective; see modeling.py / train.py."""


def __getattr__(name):
    if name == "FastFitTrainer":
        from .train import FastFitTrainer

        return FastFitTrainer
    if name in ("FastFit", "FastFitConfig", "FastFitTrainable", "ConfigArguments"):
        from . import modeling

        return getattr(modeling, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FastFit", "FastFitConfig", "FastFitTrainable", "ConfigArguments", "FastFitTrainer"]
