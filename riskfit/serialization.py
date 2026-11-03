"""Alert serialization: a vendor-agnostic normalized alert schema and the 21-field text renderer.

Every model in the paper (trained encoders, kNN, SetFit, Qwen, and the hosted models' prompts) sees an alert
as the text produced by ``render_alert``. The renderer emits at most 21 lines, in this order (a line is
omitted when its field is empty):

  1 Alert            name                       12 File               detection.file_name
  2 Severity         severity                   13 Path               detection.file_path | process_path
  3 Source           source                     14 Process Chain      process <- parent <- grandparent
  4 Category         category                   15 Command            detection.command_line
  5 Risk Score       detection.risk_score       16 Detection Engines  detection.detection_engines
  6 Obs              observables (+TI suffix)   17 Response           source_mitigation (vendor response)
  7 Objective        detection.objective        18 Trigger IOC        detection.trigger_ioc_type / _value
  8 Description      detection.description      19 Files Accessed     detection.files_accessed (first 5)
  9 Hostname         detection.hostname         20 Tactics            mitre_attack_tactics[].name
 10 Platform         detection.platform_name|os 21 Techniques         mitre_attack_techniques[].name
 11 User             detection.username

No label, analyst comment, status or prior verdict is rendered (they are envelope fields of a training
record). Verbose free-text fields are capped before rendering (``VERBOSE_FIELD_CHAR_CAPS``); the model
tokenizer truncates the final text at 1,024 tokens. The output is byte-identical to the renderer used for
the paper, which is what makes GUIDE prompt ids (sha256 of this text) reproducible.
"""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- schema


class MitreRef(BaseModel):
    model_config = ConfigDict(extra="allow")
    code: str | None = None
    name: str | None = None


class Enrichment(BaseModel):
    """One threat-intelligence provider result for an observable (raw provider payload)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)
    provider: str | None = Field(default=None, validation_alias=AliasChoices("provider", "Provider"))
    reputation: str | None = Field(default=None, validation_alias=AliasChoices("reputation", "Reputation"))
    payload: dict[str, Any] = Field(default_factory=dict, validation_alias=AliasChoices("payload", "Payload"))


class Observable(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    value: str
    description: str = ""
    reputation: str | None = None
    scope: list[str] = Field(default_factory=list)
    enrichments: list[Enrichment] = Field(default_factory=list)
    # Pre-flattened enrichment with the renderer's keys (malicious/suspicious/harmless/undetected,
    # threat_label, gti_verdict, reputation, gti_threat_score); used when ``enrichments`` is empty.
    enrichment: dict[str, Any] = Field(default_factory=dict)


class Detection(BaseModel):
    model_config = ConfigDict(extra="allow")
    risk_score: float | int | None = None
    objective: str | None = None
    description: str | None = None
    hostname: str | None = None
    os: str | None = None
    platform_name: str | None = None
    username: str | None = None
    user_domain: str | None = None
    local_ip: str | None = None
    mac_address: str | None = None
    file_name: str | None = None
    file_path: str | None = None
    process_name: str | None = None
    process_path: str | None = None
    parent_process_name: str | None = None
    grandparent_process_name: str | None = None
    command_line: str | None = None
    detection_engines: list[str] | None = None
    trigger_ioc_type: str | None = None
    trigger_ioc_value: str | None = None
    files_accessed: list[Any] | None = None


class NormalizedAlert(BaseModel):
    """The alert schema the models are trained on. Unknown fields are accepted and ignored."""

    model_config = ConfigDict(extra="allow")
    id: str | None = None
    source: str | None = None
    name: str | None = None
    category: str | None = None
    source_mitigation: str | None = None
    severity: str | None = None
    source_severity: str | None = None
    detected_at: str | None = None
    created_at: str | None = None
    detection: Detection = Field(default_factory=Detection)
    detection_type: str | None = None
    mitre_attack_tactics: list[MitreRef] = Field(default_factory=list)
    mitre_attack_techniques: list[MitreRef] = Field(default_factory=list)
    observables: list[Observable] = Field(default_factory=list)


# Rendered payload field <- ordered detection source candidates (first non-empty wins).
DETECTION_PAYLOAD_MAP: dict[str, list[str]] = {
    "risk_score": ["risk_score"],
    "objective": ["objective"],
    "description": ["description"],
    "hostname": ["hostname"],
    "platform_name": ["platform_name", "os"],
    "username": ["username"],
    "file_name": ["file_name"],
    "file_path": ["file_path", "process_path"],
    "process_name": ["process_name"],
    "parent_process_name": ["parent_process_name"],
    "grandparent_process_name": ["grandparent_process_name"],
    "command_line": ["command_line"],
    "detection_engines": ["detection_engines"],
    "trigger_ioc_type": ["trigger_ioc_type"],
    "trigger_ioc_value": ["trigger_ioc_value"],
    "files_accessed": ["files_accessed"],
}

VERBOSE_FIELD_CHAR_CAPS: dict[str, int] = {
    "command_line": 256,
    "description": 512,
    "objective": 256,
    "response_description": 256,
    "file_path": 256,
    "file_name": 160,
}

# --------------------------------------------------------------------------- enrichment flattening

_AV_KEYS = ("malicious", "suspicious", "harmless", "undetected")


def _dig(m: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(m, dict):
            return None
        m = m.get(k)
    return m


def _as_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None


def _as_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    return float(v) if isinstance(v, (int, float)) else None


def _first_network_entry(payload: dict) -> dict | None:
    entries = _dig(payload, "network_reputation", "rl", "entries")
    return entries[0] if isinstance(entries, list) and entries and isinstance(entries[0], dict) else None


def _flat_reversinglabs(payload: dict, reputation: str | None) -> dict:
    e: dict[str, Any] = {}
    presence = _dig(payload, "file_reputation", "rl", "malware_presence") or {}
    count = _as_int(presence.get("scanner_count"))
    entry = _first_network_entry(payload)
    if count and count > 0:
        match = _as_int(presence.get("scanner_match")) or 0
        e["malicious"], e["undetected"] = match, max(count - match, 0)
    elif entry is not None and isinstance(entry.get("third_party_reputations"), dict):
        tpr = entry["third_party_reputations"]
        for out_key, in_key in (("malicious", "malicious"), ("suspicious", "suspicious"),
                                ("harmless", "clean"), ("undetected", "undetected")):
            v = _as_int(tpr.get(in_key))
            if v is not None:
                e[out_key] = v
    label = presence.get("threat_name") or (entry.get("threat_name") if entry else None)
    if label:
        e["threat_label"] = label
    if reputation and reputation not in ("Unknown", ""):
        e["reputation"] = reputation
    return e


def _flat_gti(payload: dict, reputation: str | None) -> dict:
    attrs = _dig(payload, "data", "attributes") or {}
    e: dict[str, Any] = {}
    stats_src = attrs.get("last_analysis_stats")
    if isinstance(stats_src, dict):
        stats = {k: _as_int(stats_src.get(k)) for k in _AV_KEYS if _as_int(stats_src.get(k)) is not None}
        if stats:
            e["last_analysis_stats"] = stats
    rep = _as_int(attrs.get("reputation"))
    if rep is not None:
        e["reputation"] = rep
    label = _dig(attrs, "popular_threat_classification", "suggested_threat_label")
    if label:
        e["threat_label"] = label
    verdict = _dig(attrs, "gti_assessment", "verdict", "value")
    if verdict:
        e["gti_verdict"] = verdict
    score = _as_float(_dig(attrs, "gti_assessment", "threat_score", "value"))
    if score is not None:
        e["gti_threat_score"] = score
    return e


PROVIDER_FLATTENERS = {"ReversingLabs": _flat_reversinglabs, "GTI": _flat_gti}


def merge_enrichments(enrichments: list[Enrichment]) -> dict[str, Any]:
    """Flatten each provider payload and merge (first non-empty value per key wins)."""
    merged: dict[str, Any] = {}
    for enr in enrichments:
        fn = PROVIDER_FLATTENERS.get(enr.provider or "")
        if fn is None or not isinstance(enr.payload, dict):
            continue
        for k, v in fn(enr.payload, enr.reputation).items():
            merged.setdefault(k, v)
    return merged


def _observable_enrichment(obs: Observable) -> dict[str, Any]:
    if obs.enrichments:
        merged = merge_enrichments(obs.enrichments)
        if merged:
            return merged
    if obs.enrichment:
        return dict(obs.enrichment)
    if isinstance(obs.reputation, str) and obs.reputation and obs.reputation != "Unknown":
        return {"reputation": obs.reputation}
    return {}


# --------------------------------------------------------------------------- projection


def to_selected(alert: NormalizedAlert) -> dict[str, Any]:
    """Project the alert onto the renderer's intermediate (meta / payload / observables)."""
    meta = {k: v for k, v in (("name", alert.name), ("category", alert.category), ("severity", alert.severity),
                              ("source", alert.source)) if v}
    det = alert.detection.model_dump()
    payload: dict[str, Any] = {}
    for target, sources in DETECTION_PAYLOAD_MAP.items():
        for src in sources:
            value = det.get(src)
            if value not in (None, ""):
                payload[target] = value
                break
    if alert.source_mitigation:
        payload.setdefault("response_description", alert.source_mitigation)
    tactics = [t.name for t in alert.mitre_attack_tactics if t.name]
    if tactics:
        payload.setdefault("tactic_names", tactics)
    techniques = [t.name for t in alert.mitre_attack_techniques if t.name]
    if techniques:
        payload.setdefault("technique_names", techniques)
    for f, cap in VERBOSE_FIELD_CHAR_CAPS.items():
        v = payload.get(f)
        if isinstance(v, str) and len(v) > cap:
            payload[f] = v[:cap] + "…[truncated]"
    observables = [{"type": o.type, "value": o.value, "description": o.description,
                    "enrichment": _observable_enrichment(o)} for o in alert.observables]
    return {"alert_meta": meta, "alert_payload": payload, "observables": observables}


# --------------------------------------------------------------------------- renderer

_OBS_TYPE_SHORT = {"ip_address": "ip", "mac_address": "mac", "user_name": "user", "file_path": "path",
                   "process_name": "proc", "command_line": "cmd", "registry_key": "reg", "hostname": "host",
                   "certificate": "cert"}


def _present(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (dict, list, set, tuple)):
        return bool(v)
    return True


def _clip(s: str, n: int) -> str:
    return s if not s or len(s) <= n else s[:n] + "..."


def _shorten_hash(h: str) -> str:
    return h if not h or len(h) < 24 else f"{h[:12]}...{h[-8:]}"


def _shorten_path(p: str) -> str:
    if not p:
        return p
    if p.startswith("\\Device\\"):
        parts = p.split("\\", 2)
        p = parts[2] if len(parts) > 2 else p
    parts = p.split("\\") if "\\" in p else p.split("/")
    return ".../" + "/".join(parts[-2:]) if len(parts) > 2 else p


def _enrichment_suffix(e: dict[str, Any]) -> str:
    if not e:
        return ""
    parts = []
    av = [e.get(k) for k in _AV_KEYS]
    vals = [v for v in av if v is not None]
    if vals and sum(vals) > 0:
        parts.append(f"AV:{av[0] if av[0] is not None else 0}/{sum(vals)}")
    for key in ("threat_label", "gti_verdict", "reputation", "gti_threat_score"):
        v = e.get(key)
        if v is None or (isinstance(v, str) and not v) or isinstance(v, (dict, list)):
            continue
        parts.append(f"{key}={_clip(str(v), 80)}")
    sv = e.get("sandbox_verdicts")
    if isinstance(sv, dict):
        vp = [f"{vendor}:{v['category']}" for vendor, v in sv.items() if isinstance(v, dict) and v.get("category")][:3]
        if vp:
            if len(sv) - len(vp) > 0:
                vp.append(f"+{len(sv) - len(vp)}")
            parts.append(f"vendor={','.join(vp)}")
    return f"({'; '.join(parts)})" if parts else ""


def render_selected(selected: dict[str, Any]) -> str:
    meta, pl, obs = selected.get("alert_meta", {}), selected.get("alert_payload", {}), selected.get("observables", [])
    out: list[str] = []

    def add(label: str, v: Any) -> None:
        if _present(v):
            out.append(f"{label}: {v}")

    add("Alert", meta.get("name"))
    add("Severity", meta.get("severity"))
    add("Source", meta.get("source"))
    add("Category", meta.get("category"))
    if _present(pl.get("risk_score")):
        out.append(f"Risk Score: {pl['risk_score']}")
    if obs:
        seen, lines = set(), []
        for o in obs:
            key = (o.get("type"), o.get("value"))
            if key in seen:
                continue
            seen.add(key)
            raw = o.get("value", "")
            if not raw or set(raw) == {"0"}:  # empty or all-zero placeholder
                continue
            value = o.get("value", "N/A")
            t = str(o.get("type", "unknown")).lower().replace("observable_type_", "")
            t = _OBS_TYPE_SHORT.get(t, t[:6])
            if t == "hash":
                value = _shorten_hash(value)
            elif t == "path":
                value = _shorten_path(value)
            elif t == "cmd":
                value = _clip(value, 80)
            elif t == "url":
                value = _clip(value, 60)
            lines.append(f"{t}:{value}{_enrichment_suffix(o.get('enrichment', {}))}")
        if lines:
            out.append("Obs: " + " | ".join(lines))
    add("Objective", pl.get("objective"))
    add("Description", pl.get("description"))
    add("Hostname", pl.get("hostname"))
    add("Platform", pl.get("platform_name"))
    add("User", pl.get("username"))
    add("File", pl.get("file_name"))
    add("Path", pl.get("file_path"))
    if _present(pl.get("process_name")):
        chain = [pl["process_name"]]
        for k in ("parent_process_name", "grandparent_process_name"):
            if _present(pl.get(k)):
                chain.append(pl[k])
        out.append(f"Process Chain: {' <- '.join(chain)}")
    add("Command", pl.get("command_line"))
    engines = pl.get("detection_engines")
    if isinstance(engines, list) and engines:
        engines = [str(x) for x in engines if _present(x)]
        if engines:
            out.append(f"Detection Engines: {', '.join(engines)}")
    elif isinstance(engines, str) and engines.strip():
        out.append(f"Detection Engine: {engines}")
    add("Response", pl.get("response_description"))
    it, iv = pl.get("trigger_ioc_type", ""), pl.get("trigger_ioc_value", "")
    if _present(it) and _present(iv):
        out.append(f"Trigger IOC: {it} = {iv}")
    elif _present(it):
        out.append(f"IOC Type: {it}")
    elif _present(iv):
        out.append(f"IOC Value: {iv}")
    files = pl.get("files_accessed")
    if isinstance(files, list) and files:
        names = []
        for f in files[:5]:
            if isinstance(f, dict) and _present(f.get("file_name", "")):
                names.append(f["file_name"])
            elif isinstance(f, str) and f.strip():
                names.append(f)
        if names:
            s = ", ".join(names) + (f" (+{len(files) - 5} more)" if len(files) > 5 else "")
            out.append(f"Files Accessed: {s}")
    elif isinstance(files, str) and files.strip():
        out.append(f"Files Accessed: {files}")
    for key, label in (("tactic_names", "Tactics"), ("technique_names", "Techniques")):
        if pl.get(key):
            vals = [str(x) for x in pl[key] if _present(x)]
            if vals:
                out.append(f"{label}: {', '.join(vals)}")
    return "\n\n".join(out)


def render_alert(record: dict[str, Any]) -> str:
    """Render one alert record (normalized schema) to model text."""
    alert = NormalizedAlert.model_validate(record)
    return render_selected(to_selected(alert))
