"""Non-adaptive evasion perturbations of attacker-controllable alert fields (deterministic, seed 42).

Generic functions over normalized alert records (``riskfit.serialization`` schema). Used in the paper on the
private production test alerts (RQ2 "Evasive alerts"); no private data is included here.

Applied to the RAW alert record (before any model's renderer), so every model sees the same perturbed alert
through its own unchanged rendering path.

Never touched (by any perturbation): alert name, severity, source/vendor, category, tenant, labels, IDs,
timestamps, MITRE tactic/technique labels, detection-engine list, the detection rule name, and (except R0)
the vendor response fields.

P1  benign-name masquerade: every executable/script/installer basename (an executable extension; on non-Windows
    alerts also extension-less binary names) that is NOT on the OS/LOLBin/browser allow-list (KEEP_NAMES) is renamed to a common benign name of the same kind (same extension; macOS/Linux
    daemon names for extension-less POSIX binaries). The new name is substituted consistently in every
    substitution-eligible string (process/file names and paths, command lines, observables, description,
    trigger IOC). Directories and file hashes are unchanged.
P2  command-line obfuscation (semantics-preserving): every command line whose program is not an OS service
    process (SERVICE_PROCS) is rewritten:
      PowerShell            -> same program (random case) + kept flags + -EncodedCommand <base64 UTF-16LE script>
      cmd.exe /c|/k         -> random-case program, caret (^) inserted before letters (p=0.3) outside quotes/%vars%
      POSIX shell -c        -> <shell> -c "$(echo <base64 script> | base64 -d)"
      other Windows program -> powershell.exe -NoP -NonI -W Hidden -EncodedCommand <base64 of "& '<argv0>' <args>">
      other POSIX program   -> sh -c "$(echo <base64 original> | base64 -d)"
    Already-encoded PowerShell keeps its payload (only program/flag case is randomized).
P3  fresh infrastructure: every IOC-type observable (public IP, domain/hostname, URL, hash, e-mail) that carries a
    threat-intelligence hit (AV/engine malicious>0 or suspicious>0, reputation Malicious/Suspicious, a threat
    label, or a malicious/suspicious GTI verdict) gets a fresh value of the same format (new public IP, new
    domain, same-length hex hash) and all its enrichment (enrichments, enrichment, reputation) is removed.
    The new value is substituted consistently everywhere. Private/loopback IPs and clean observables are kept.
P4  P3 then P1 then P2.
R0  (leakage control, not an attack) remove the vendor Response field (top-level source_mitigation).
"""
from __future__ import annotations

import base64
import copy
import hashlib
import ipaddress
import random
import re
import shlex
from typing import Any

SEED = 42
PERTURBATIONS = ("P1", "P2", "P3", "P4", "R0")

# keys never modified (alert level and detection level)
FROZEN_KEYS = {
    "name", "severity", "source_severity", "source", "category", "label", "label_source", "tenant", "tenant_id",
    "id", "alert_id", "created_at", "detected_at", "mitre_attack_tactics", "mitre_attack_techniques",
    "detection_type", "source_mitigation", "vendor_name", "product_name", "detection_engines", "tactics",
    "tactic_names", "techniques", "technique_names", "pattern_id", "detection_id", "vendor_alert_id",
    "aggregate_id", "customer_id", "agent_id", "console_link", "response_description", "actions_taken",
    "response_flags", "platform_name", "platform_id", "os", "type", "custom_rule_name", "custom_rule_group",
}

# P1 allow-list: OS binaries, LOLBins/interpreters, major browsers (lower-case basenames) -> never renamed
KEEP_NAMES = {
    # Windows OS / services
    "svchost.exe", "services.exe", "explorer.exe", "dllhost.exe", "userinit.exe", "winlogon.exe", "sihost.exe",
    "lsass.exe", "wmiprvse.exe", "smss.exe", "csrss.exe", "wininit.exe", "runtimebroker.exe", "taskhostw.exe",
    "conhost.exe", "spoolsv.exe", "searchindexer.exe", "werfault.exe", "taskmgr.exe", "ctfmon.exe", "lsm.exe",
    "audiodg.exe", "fontdrvhost.exe", "dwm.exe", "smartscreen.exe", "msmpeng.exe", "mpcmdrun.exe",
    # LOLBins / interpreters / admin tools
    "cmd.exe", "powershell.exe", "pwsh.exe", "powershell_ise.exe", "rundll32.exe", "regsvr32.exe", "mshta.exe",
    "wscript.exe", "cscript.exe", "msiexec.exe", "certutil.exe", "bitsadmin.exe", "schtasks.exe", "sc.exe",
    "net.exe", "net1.exe", "reg.exe", "curl.exe", "wget.exe", "tar.exe", "whoami.exe", "nltest.exe",
    "wmic.exe", "forfiles.exe", "hh.exe", "installutil.exe", "regasm.exe", "regsvcs.exe", "msbuild.exe",
    "python.exe", "pythonw.exe", "py.exe", "node.exe", "java.exe", "javaw.exe", "ssh.exe", "scp.exe",
    "notepad.exe", "mmc.exe", "control.exe", "rundll.exe", "at.exe", "ftp.exe", "nslookup.exe", "ping.exe",
    # browsers / mail / office
    "chrome.exe", "msedge.exe", "iexplore.exe", "firefox.exe", "outlook.exe", "winword.exe", "excel.exe",
    "powerpnt.exe", "teams.exe", "ms-teams.exe", "onedrive.exe",
    # POSIX
    "sh", "bash", "zsh", "dash", "ksh", "fish", "curl", "wget", "python", "python3", "perl", "ruby", "node",
    "osascript", "launchd", "runningboardd", "sudo", "ssh", "scp", "nc", "ncat", "tar", "zip", "unzip",
    "base64", "openssl", "chmod", "chown", "cp", "mv", "rm", "cat", "echo", "env", "nohup", "kill", "ps",
    "systemd", "cron", "crond", "sshd", "login", "su",
}
EXEC_EXT = {".exe", ".dll", ".scr", ".msi", ".ps1", ".psm1", ".bat", ".cmd", ".vbs", ".vbe", ".js", ".jse",
            ".hta", ".tmp", ".sh", ".py", ".jar", ".app", ".com", ".cpl", ".lnk", ".sys", ".bin", ".elf",
            ".dylib", ".so", ".pkg", ".dmg", ".appimage", ".deb", ".rpm"}
BENIGN = {
    ".exe": ["svchost.exe", "RuntimeBroker.exe", "taskhostw.exe", "OneDriveStandaloneUpdater.exe",
             "MicrosoftEdgeUpdate.exe", "GoogleUpdate.exe", "AdobeARM.exe"],
    ".dll": ["version.dll", "msvcp140.dll", "vcruntime140.dll"],
    ".msi": ["GoogleChromeStandaloneEnterprise64.msi", "OneDriveSetup.msi", "AcroRdrDCUpd.msi"],
    ".ps1": ["Update-DefenderSignatures.ps1", "Invoke-IntuneHealthCheck.ps1"],
    "posix_mac": ["softwareupdated", "mdworker_shared", "trustd", "nsurlsessiond"],
    "posix_linux": ["systemd-resolved", "dbus-daemon", "snapd", "unattended-upgrade"],
    "mac_helper": ["Google Chrome Helper (Renderer)", "Microsoft Teams Helper (Renderer)"],
    "stems": ["MicrosoftEdgeUpdate", "GoogleUpdate", "OneDriveUpdater", "AdobeARM", "RuntimeBroker"],
}
SERVICE_PROCS = {"svchost.exe", "services.exe", "userinit.exe", "winlogon.exe", "explorer.exe", "dllhost.exe",
                 "sihost.exe", "lsass.exe", "wmiprvse.exe", "smss.exe", "csrss.exe", "wininit.exe",
                 "runtimebroker.exe", "taskhostw.exe", "conhost.exe", "runningboardd", "launchd"}
CMD_FIELDS = ("command_line", "parent_command_line", "grandparent_command_line")
NAME_FIELDS = ("process_name", "file_name", "parent_process_name", "grandparent_process_name")
PATH_FIELDS = ("process_path", "file_path", "parent_process_path", "grandparent_process_path")
IOC_TYPES = {"ip_address", "ip", "ipv4", "ipv6", "domain", "hostname", "url", "hash", "file_hash", "sha256",
             "sha1", "md5", "email_address", "email"}
WORDS = ["cloud", "sync", "portal", "data", "share", "edge", "cdn", "secure", "update", "files", "static",
         "assets", "api", "drive", "media", "link", "hub", "app", "store", "net", "box", "web", "go", "dl"]
TLDS = ["com", "net", "io", "org", "co"]


def rng_for(*parts: Any) -> random.Random:
    h = hashlib.sha256("|".join([str(SEED), *map(str, parts)]).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


# ----------------------------------------------------------------------------- generic helpers

def walk_strings(obj: Any, fn, key: str | None = None, frozen=FROZEN_KEYS):
    """Apply fn to every string value whose key is not frozen (recursively)."""
    if isinstance(obj, dict):
        return {k: (v if k in frozen else walk_strings(v, fn, k, frozen)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [walk_strings(v, fn, key, frozen) for v in obj]
    if isinstance(obj, str):
        return fn(obj)
    return obj


def substitute(rec: dict, mapping: dict[str, str], case_insensitive=True) -> dict:
    """Consistent token substitution in all substitution-eligible strings (observable types stay frozen)."""
    if not mapping:
        return rec
    keys = sorted(mapping, key=len, reverse=True)
    flags = re.IGNORECASE if case_insensitive else 0
    pat = re.compile("|".join(rf"(?<![\w-]){re.escape(k)}(?![\w-])" for k in keys), flags)
    low = {(k.lower() if case_insensitive else k): v for k, v in mapping.items()}

    def fn(s):
        return pat.sub(lambda m: low[m.group(0).lower() if case_insensitive else m.group(0)], s)

    return walk_strings(rec, fn)


def basename(p: str) -> str:
    return re.split(r"[\\/]", p.strip().strip('"').strip("'"))[-1]


def is_windows(rec: dict) -> bool:
    d = rec.get("detection") or {}
    plat = f"{d.get('platform_name') or ''} {d.get('os') or ''}".lower()
    if "win" in plat:
        return True
    if "mac" in plat or "linux" in plat or "darwin" in plat:
        return False
    blob = " ".join(str(o.get("value", "")) for o in rec.get("observables") or [])
    blob += " " + " ".join(str(d.get(f) or "") for f in CMD_FIELDS + PATH_FIELDS + NAME_FIELDS)
    return bool(re.search(r"\.exe\b|[A-Za-z]:\\", blob, re.I))


_MAC_PATH = re.compile(r"/(?:Users|Applications)/")  # macOS home and application directories


def is_mac(rec: dict) -> bool:
    blob = str(rec.get("observables")) + str(rec.get("detection"))
    return bool(_MAC_PATH.search(blob)) or "macos" in blob.lower()


def split_argv0(cmd: str) -> tuple[str, str]:
    s = cmd.lstrip()
    if s.startswith('"'):
        j = s.find('"', 1)
        if j > 0:
            return s[: j + 1], s[j + 1:]
    m = re.match(r"(\S+)(.*)$", s, re.S)
    return (m.group(1), m.group(2)) if m else (s, "")


def randcase(s: str, rng: random.Random) -> str:
    return "".join(c.upper() if (c.isalpha() and rng.random() < 0.5) else (c.lower() if c.isalpha() else c) for c in s)


# ----------------------------------------------------------------------------- P1

def candidate_names(rec: dict) -> set[str]:
    d = rec.get("detection") or {}
    names = set()
    for f in NAME_FIELDS:
        if d.get(f):
            names.add(basename(d[f]))
    for f in PATH_FIELDS:
        if d.get(f):
            names.add(basename(d[f]))
    for f in CMD_FIELDS:
        if d.get(f):
            names.add(basename(split_argv0(d[f])[0]))
    for o in rec.get("observables") or []:
        v = str(o.get("value") or "")
        if o.get("type") in ("process_name", "file_name", "file_path", "process_path"):
            names.add(basename(v))
        elif o.get("type") == "command_line":
            names.add(basename(split_argv0(v)[0]))
    out = set()
    win = is_windows(rec)
    for n in names:
        n = n.strip()
        if not n or n.lower() in KEEP_NAMES:
            continue
        ext = ("." + n.rsplit(".", 1)[1].lower()) if "." in n else ""
        # extension-less names count as (POSIX) binaries only on non-Windows alerts (on Windows they are directories)
        if ext in EXEC_EXT or (ext == "" and not win and not re.fullmatch(r"[\d.]+", n)):
            out.add(n)
    return out


def benign_name(name: str, rec: dict) -> str:
    r = rng_for("P1", rec.get("alert_id"), name.lower())
    ext = ("." + name.rsplit(".", 1)[1]) if "." in name and " " not in name.rsplit(".", 1)[1] else ""
    e = ext.lower()
    if e in BENIGN:
        new = r.choice(BENIGN[e])
        return new[: -len(e)] + ext if new.lower().endswith(e) else new
    if e:
        return r.choice(BENIGN["stems"]) + ext
    if " " in name:
        return r.choice(BENIGN["mac_helper"])
    return r.choice(BENIGN["posix_mac"] if is_mac(rec) else BENIGN["posix_linux"])


def p1(rec: dict) -> dict:
    names = candidate_names(rec)
    mapping = {n: benign_name(n, rec) for n in names}
    # Windows-style names (with an extension) match case-insensitively; extension-less POSIX names case-sensitively
    rec = substitute(rec, {k: v for k, v in mapping.items() if "." in k}, case_insensitive=True)
    return substitute(rec, {k: v for k, v in mapping.items() if "." not in k}, case_insensitive=False)


# ----------------------------------------------------------------------------- P2

def b64_utf16(s: str) -> str:
    return base64.b64encode(s.encode("utf-16-le")).decode()


def b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


PS_KEEP_FLAGS = {"-noprofile", "-nop", "-noninteractive", "-noni", "-nologo", "-sta", "-mta", "-windowstyle",
                 "-w", "-executionpolicy", "-ep", "-exec", "-ex"}
PS_FLAGS_WITH_ARG = {"-windowstyle", "-w", "-executionpolicy", "-ep", "-exec", "-ex"}


def obf_powershell(argv0: str, rest: str, rng) -> str:
    try:
        toks = shlex.split(rest, posix=False)
    except ValueError:
        toks = rest.split()
    low = [t.lower() for t in toks]
    if any(t in ("-enc", "-e", "-ec", "-encodedcommand") for t in low):
        flags = [randcase(t, rng) if t.startswith("-") else t for t in toks]
        return f"{randcase(argv0, rng)} {' '.join(flags)}"
    kept, i, script = [], 0, None
    while i < len(toks):
        t = low[i]
        if t in ("-command", "-c"):
            script = " ".join(toks[i + 1:]); break
        if t in ("-file", "-f"):
            path = toks[i + 1].strip('"') if i + 1 < len(toks) else ""
            args = " ".join(toks[i + 2:])
            script = f"& '{path}' {args}".rstrip(); break
        if t in PS_KEEP_FLAGS:
            kept.append(toks[i])
            if t in PS_FLAGS_WITH_ARG and i + 1 < len(toks):
                kept.append(toks[i + 1]); i += 1
            i += 1; continue
        script = " ".join(toks[i:]); break
    if not script:
        return f"{randcase(argv0, rng)} {' '.join(kept)}".rstrip()
    flags = " ".join(randcase(k, rng) if k.startswith("-") else k for k in kept)
    return f"{randcase(argv0, rng)} {flags + ' ' if flags else ''}{randcase('-EncodedCommand', rng)} {b64_utf16(script)}"


def caret_insert(s: str, rng) -> str:
    out, inq, inpct = [], False, False
    for c in s:
        if c == '"':
            inq = not inq
        elif c == "%" and not inq:
            inpct = not inpct
        if c.isalpha() and not inq and not inpct and rng.random() < 0.3:
            out.append("^")
        out.append(c)
    return "".join(out)


def obf_cmdline(cmd: str, rec: dict) -> str:
    if not cmd or not cmd.strip():
        return cmd
    rng = rng_for("P2", rec.get("alert_id"), cmd)
    argv0, rest = split_argv0(cmd)
    prog = basename(argv0).lower()
    if prog in SERVICE_PROCS:
        return cmd
    win = is_windows(rec) or bool(re.search(r"\.exe\b|^[A-Za-z]:\\", argv0, re.I))
    if prog in ("powershell.exe", "pwsh.exe", "powershell", "pwsh"):
        return obf_powershell(argv0, rest, rng)
    if prog in ("cmd.exe", "cmd"):
        m = re.match(r"(\s*)(/[ckCK])(\s+)(.*)$", rest, re.S)
        if m:
            script = m.group(4)
            if script.startswith('"') and script.count('"') >= 2:
                # cmd /c strips the first and the last quote before parsing: insert carets w.r.t. the stripped text
                last = script.rfind('"')
                body = '"' + caret_insert(script[1:last], rng) + '"' + script[last + 1:]
            else:
                body = caret_insert(script, rng)
            return f"{randcase(argv0, rng)}{m.group(1)}{m.group(2).upper()}{m.group(3)}{body}"
        return f"{randcase(argv0, rng)}{caret_insert(rest, rng)}"
    if prog in ("sh", "bash", "zsh", "dash", "ksh"):
        m = re.search(r"(?:^|\s)-c\s+(.*)$", rest, re.S)
        if m:
            pre = rest[: m.start(1)]
            script = m.group(1).strip()
            if len(script) >= 2 and script[0] == script[-1] and script[0] in "'\"":
                script = script[1:-1]
            return f'{argv0}{pre}"$(echo {b64(script)} | base64 -d)"'
    if win:
        a0 = argv0.strip('"')
        inner = f"& '{a0}' {rest.strip()}".rstrip()
        return f"powershell.exe -NoP -NonI -W Hidden -EncodedCommand {b64_utf16(inner)}"
    return f'sh -c "$(echo {b64(cmd)} | base64 -d)"'


def p2(rec: dict) -> dict:
    rec = copy.deepcopy(rec)
    d = rec.get("detection") or {}
    for f in CMD_FIELDS:
        if isinstance(d.get(f), str) and d[f]:
            d[f] = obf_cmdline(d[f], rec)
    for o in rec.get("observables") or []:
        if o.get("type") == "command_line" and isinstance(o.get("value"), str):
            o["value"] = obf_cmdline(o["value"], rec)
    return rec


# ----------------------------------------------------------------------------- P3

def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def ti_hit(o: dict) -> bool:
    blobs = []
    for e in o.get("enrichments") or []:
        if isinstance(e, dict):
            blobs.append(e.get("payload") or {}); blobs.append({"reputation": e.get("reputation") or e.get("Reputation")})
    if isinstance(o.get("enrichment"), dict):
        blobs.append(o["enrichment"])
    blobs.append({"reputation": o.get("reputation")})

    def scan(x) -> bool:
        if isinstance(x, dict):
            for k, v in x.items():
                kl = str(k).lower()
                if kl in ("malicious", "suspicious", "positives") and _num(v) > 0:
                    return True
                if kl in ("reputation", "verdict", "gti_verdict", "threat_verdict", "classification") and isinstance(v, str) \
                        and re.search(r"malicious|suspicious", v, re.I):
                    return True
                if kl in ("threat_label", "threat_name", "malware_family") and v:
                    return True
                if scan(v):
                    return True
        elif isinstance(x, list):
            return any(scan(v) for v in x)
        return False

    return any(scan(b) for b in blobs)


def fresh_ip(old: str, rng) -> str:
    try:
        v6 = ipaddress.ip_address(old).version == 6
    except ValueError:
        v6 = False
    while True:
        if v6:
            ip = ipaddress.IPv6Address(rng.getrandbits(128) | (0x2 << 124))
        else:
            ip = ipaddress.IPv4Address(rng.getrandbits(32))
        if ip.is_global and not ip.is_multicast:
            return str(ip)


def fresh_domain(rng) -> str:
    return f"{rng.choice(WORDS)}{rng.choice(['', '-'])}{rng.choice(WORDS)}{rng.randint(1, 99) if rng.random() < .3 else ''}.{rng.choice(TLDS)}"


def p3(rec: dict) -> dict:
    rec = copy.deepcopy(rec)
    aid = rec.get("alert_id")
    mapping: dict[str, str] = {}
    for o in rec.get("observables") or []:
        t = str(o.get("type") or "").lower()
        v = str(o.get("value") or "")
        if t not in IOC_TYPES or not v or not ti_hit(o):
            continue
        rng = rng_for("P3", aid, v.lower())
        if t in ("ip_address", "ip", "ipv4", "ipv6"):
            try:
                if not ipaddress.ip_address(v).is_global:
                    continue
            except ValueError:
                continue
            mapping[v] = fresh_ip(v, rng)
        elif t in ("hash", "file_hash", "sha256", "sha1", "md5") and re.fullmatch(r"[0-9a-fA-F]{16,}", v):
            new = "".join(rng.choice("0123456789abcdef") for _ in v)
            mapping[v] = new.upper() if v.isupper() else new
        elif t == "url":
            m = re.match(r"^([a-zA-Z][\w+.-]*://)?([^/:?#]+)(.*)$", v)
            if m:
                host = m.group(2)
                try:
                    ipaddress.ip_address(host); newhost = fresh_ip(host, rng)
                except ValueError:
                    newhost = fresh_domain(rng)
                mapping[host] = newhost
        elif t in ("domain", "hostname"):
            mapping[v] = fresh_domain(rng)
        elif t in ("email_address", "email") and "@" in v:
            mapping[v.split("@", 1)[1]] = fresh_domain(rng)
        else:
            continue
        o["enrichments"] = []
        o["enrichment"] = {}
        o["reputation"] = None
    # observable 'value' keys are not frozen, so the new values propagate into observables too
    return substitute(rec, mapping)


# ----------------------------------------------------------------------------- R0 / P4 / dispatcher

def r0(rec: dict) -> dict:
    rec = copy.deepcopy(rec)
    rec.pop("source_mitigation", None)
    return rec


def apply(rec: dict, which: str) -> dict:
    if which == "U":
        return copy.deepcopy(rec)
    if which == "P1":
        return p1(rec)
    if which == "P2":
        return p2(rec)
    if which == "P3":
        return p3(rec)
    if which == "P4":
        return p2(p1(p3(rec)))
    if which == "R0":
        return r0(rec)
    raise ValueError(which)
