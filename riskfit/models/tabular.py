"""LightGBM on structured alert fields (the paper's tree baseline).

Features are read only from the normalized alert schema (never from the label/provenance envelope):
severity, name keywords, detection time, risk score, path/command traits, observable type counts,
hash/IP shapes, malicious-reputation counts (this is how threat-intelligence enrichment enters), and MITRE
tactic codes.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import numpy as np

_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def _sha_like(s: str) -> tuple[int, int, int]:
    if not s or not isinstance(s, str) or not _HEX_RE.match(s):
        return 0, 0, 0
    return (int(len(s) == 32), int(len(s) == 40), int(len(s) == 64))


def _is_ipv4(s: str) -> bool:
    if not isinstance(s, str):
        return False
    parts = s.split(".")
    if len(parts) != 4:
        return False
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return False
    return all(0 <= n <= 255 for n in nums)


def _is_private_ipv4(s: str) -> bool:
    if not _is_ipv4(s):
        return False
    a, b, _, _ = [int(p) for p in s.split(".")]
    return a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168)


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _count_separators(path: str) -> int:
    return path.count("\\") + path.count("/") if isinstance(path, str) else 0


def _file_ext(path: str) -> str:
    if not isinstance(path, str):
        return ""
    base = path.split("\\")[-1].split("/")[-1]
    return base.rsplit(".", 1)[-1].lower() if "." in base else ""


class TabularFeatureExtractor:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.top_observable_types: list[str] = self.config.get(
            "top_observable_types", ["hash", "ip_address", "file_path", "hostname", "process_name", "command_line"])

    def extract(self, record: dict[str, Any]) -> dict[str, float]:
        det = record.get("detection") if isinstance(record.get("detection"), dict) else {}
        obs = record.get("observables") if isinstance(record.get("observables"), list) else []
        tactics = record.get("mitre_attack_tactics") if isinstance(record.get("mitre_attack_tactics"), list) else []
        detected = _parse_ts(record.get("detected_at") if isinstance(record.get("detected_at"), str) else None)
        f: dict[str, float] = {}
        f.update(self._alert(record, detected))
        f.update(self._detection(det))
        f.update(self._observables(obs))
        f.update(self._mitre(tactics))
        return f

    @staticmethod
    def _alert(record, detected):
        severity = str(record.get("severity", "") or "").lower()
        smap = {"informational": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        name = str(record.get("name", "") or "")
        nl = name.lower()
        f = {
            "normalized_severity_ordinal": float(smap.get(severity, -1)),
            "normalized_name_length": float(len(name)),
            "normalized_name_has_powershell": 1.0 if "powershell" in nl else 0.0,
            "normalized_name_has_ransom": 1.0 if "ransom" in nl else 0.0,
            "normalized_name_has_pup": 1.0 if "potentially unwanted" in nl or "pup" in nl else 0.0,
            "normalized_has_source": 1.0 if str(record.get("source", "") or "") else 0.0,
            "normalized_detected_hour": float(detected.hour) if detected else -1.0,
            "normalized_detected_day_of_week": float(detected.weekday()) if detected else -1.0,
        }
        for label in smap:
            f[f"normalized_severity_is_{label}"] = 1.0 if severity == label else 0.0
        return f

    @staticmethod
    def _detection(det):
        rs = det.get("risk_score")
        cmd = str(det.get("command_line", "") or "")
        cl = cmd.lower()
        fp = str(det.get("file_path") or det.get("process_path") or "")
        fl = fp.lower()
        os_ = str(det.get("os", "") or "").lower()
        return {
            "normalized_risk_score": float(rs) if isinstance(rs, int | float) else 0.0,
            "normalized_has_hostname": 1.0 if det.get("hostname") else 0.0,
            "normalized_has_username": 1.0 if det.get("username") else 0.0,
            "normalized_has_file_path": 1.0 if fp else 0.0,
            "normalized_path_length": float(len(fp)),
            "normalized_path_depth": float(_count_separators(fp)),
            "normalized_path_is_executable": 1.0 if _file_ext(fp) in {"exe", "dll", "ps1", "sh"} else 0.0,
            "normalized_path_in_temp": 1.0 if "\\temp\\" in fl or "/tmp/" in fl else 0.0,
            "normalized_has_command_line": 1.0 if cmd else 0.0,
            "normalized_command_length": float(len(cmd)),
            "normalized_command_has_encoded_flag": 1.0 if "-enc " in cl else 0.0,
            "normalized_command_has_download": 1.0 if any(
                t in cl for t in ("wget", "curl", "downloadstring", "invoke-webrequest")) else 0.0,
            "normalized_os_is_windows": 1.0 if os_ == "windows" else 0.0,
            "normalized_os_is_linux": 1.0 if os_ == "linux" else 0.0,
        }

    def _observables(self, obs):
        f = {"normalized_observable_count": float(len(obs))}
        for t in self.top_observable_types:
            f[f"normalized_observable_type_{t}_count"] = 0.0
        f["normalized_observable_other_type_count"] = 0.0
        hashes = ipv4 = private = paths = bad_rep = 0
        types: set[str] = set()
        for o in obs:
            if not isinstance(o, dict):
                continue
            t = str(o.get("type", "") or "").lower()
            v = str(o.get("value", "") or "")
            rep = str(o.get("reputation", "") or "").lower()
            if t:
                types.add(t)
            key = f"normalized_observable_type_{t}_count"
            f[key if key in f else "normalized_observable_other_type_count"] += 1.0
            hashes += sum(_sha_like(v))
            if _is_ipv4(v):
                ipv4 += 1
                private += int(_is_private_ipv4(v))
            if _count_separators(v) >= 2:
                paths += 1
            if rep in {"malicious", "suspicious"}:
                bad_rep += 1
        f.update({"normalized_observable_unique_type_count": float(len(types)),
                  "normalized_observable_hash_like_count": float(hashes),
                  "normalized_observable_ipv4_count": float(ipv4),
                  "normalized_observable_private_ipv4_count": float(private),
                  "normalized_observable_path_like_count": float(paths),
                  "normalized_observable_malicious_reputation_count": float(bad_rep)})
        return f

    @staticmethod
    def _mitre(tactics):
        codes = {str(t.get("code", "") or "").upper() for t in tactics if isinstance(t, dict) and t.get("code")}
        return {"normalized_mitre_tactic_count": float(len(codes)),
                "normalized_mitre_has_execution": 1.0 if "TA0002" in codes else 0.0,
                "normalized_mitre_has_credential_access": 1.0 if "TA0006" in codes else 0.0,
                "normalized_mitre_has_impact": 1.0 if "TA0040" in codes else 0.0}


class LightGBMModel:
    """LightGBM over feature dicts. Recipe keys other than metadata are LGBMClassifier parameters."""

    METADATA_KEYS = {"description", "model_type", "preprocessor", "preprocessor_kwargs"}

    def __init__(self, recipe: dict[str, Any]):
        params = {k: v for k, v in recipe.items() if k not in self.METADATA_KEYS}
        defaults = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": -1, "num_leaves": 63,
                    "min_child_samples": 20, "reg_alpha": 0.0, "reg_lambda": 0.0, "class_weight": None,
                    "random_state": 42}
        self.params = {**defaults, **params}
        self.extractor = TabularFeatureExtractor(recipe.get("preprocessor_kwargs") or {})
        self.model = None
        self.feature_names: list[str] = []
        self.classes_ = None

    @staticmethod
    def _matrix(features, names=None):
        keys = names if names is not None else sorted({k for f in features for k in f})
        X = np.zeros((len(features), len(keys)), dtype=np.float32)
        for i, f in enumerate(features):
            for j, k in enumerate(keys):
                try:
                    X[i, j] = float(f.get(k, 0.0))
                except Exception:
                    X[i, j] = 0.0
        return X, keys

    def fit(self, records, labels, val_records=None, val_labels=None):
        import lightgbm as lgb
        from sklearn.preprocessing import LabelEncoder

        X, self.feature_names = self._matrix([self.extractor.extract(r) for r in records])
        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(labels)
        self.classes_ = self.encoder.classes_
        eval_set = None
        if val_records:
            Xv, _ = self._matrix([self.extractor.extract(r) for r in val_records], self.feature_names)
            eval_set = [(Xv, self.encoder.transform(val_labels))]
        self.model = lgb.LGBMClassifier(**{"verbose": 0, **self.params})
        self.model.fit(X, y, eval_set=eval_set, feature_name=self.feature_names)
        return self

    def predict_proba(self, records) -> np.ndarray:
        X, _ = self._matrix([self.extractor.extract(r) for r in records], self.feature_names)
        return self.model.predict_proba(X)
