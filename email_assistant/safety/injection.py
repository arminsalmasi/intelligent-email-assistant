"""Inbound threat detection (defence in depth, not a silver bullet).

Signals are combined as independent evidence: risk = 1 - prod(1 - w_i).
A detector is only the first layer; the real protection is that the model can
only *propose* typed actions, outbound recipients are restricted, confidential
data is blocked from leaving the org, and a human approves every outgoing action.
Detected threats go to a human quarantine queue - never silently deleted.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from email_assistant.safety.untrusted import INVISIBLE, normalise
from email_assistant.schemas.email import EmailThread
from email_assistant.schemas.state import GuardResult


@dataclass(frozen=True)
class Signal:
    name: str
    weight: float
    pattern: re.Pattern[str]


_I = re.IGNORECASE | re.DOTALL
TEXT_SIGNALS = [
    Signal(
        "override_instructions",
        0.6,
        re.compile(
            r"\b(ignore|disregard|forget)\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+"
            r"(instructions?|rules|prompts?|polic(y|ies))",
            _I,
        ),
    ),
    Signal(
        "system_override",
        0.6,
        re.compile(
            r"(system\s+override|admin(istrative)?\s+(debug\s+)?mode|debug\s+mode|"
            r"safety\s+rules\s+(are\s+)?suspended|no\s+longer\s+bound\s+by)",
            _I,
        ),
    ),
    Signal(
        "role_marker",
        0.45,
        re.compile(
            r"(^|\n)\s*(system|assistant|developer)\s*:|<\s*system\s*>|\[\s*system\s*\]", _I
        ),
    ),
    # Weight < threshold on purpose: newsletters talk about "AI assistants" too;
    # it only quarantines in combination with another signal.
    Signal(
        "addresses_ai",
        0.45,
        re.compile(
            r"\b(note\s+to\s+ai|ai\s+assistants?|as\s+the\s+ai|language\s+model|dear\s+(ai|assistant)|"
            r"assistant\s*[,:]\s*(please\s+)?(forward|send|book|reply|mark))",
            _I,
        ),
    ),
    Signal(
        "hitl_bypass",
        0.5,
        re.compile(
            r"(without\s+(asking|human\s+review|approval|confirmation)|auto-?approv|"
            r"do\s+not\s+(ask|alert|mention|tell)|no\s+need\s+to\s+bother)",
            _I,
        ),
    ),
    Signal(
        "exfiltration",
        0.4,
        re.compile(
            r"\b(forward|send|email|share|attach|reply\s+with)\b.{0,80}\b(budget|figures|numbers|"
            r"contact\s+list|bank\s+(account\s+)?details|credentials|passwords?|confidential|board\s+deck)",
            _I,
        ),
    ),
    Signal(
        "decode_and_follow",
        0.5,
        re.compile(r"decode\s+(it\s+|this\s+)?and\s+(follow|execute|run)", _I),
    ),
]
HTML_COMMENT = re.compile(r"<!--(.*?)-->", re.DOTALL)
BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
URL = re.compile(r"https?://[^\s>)\"']+", re.IGNORECASE)
PHISHING_WORDS = re.compile(
    r"(reset\s+(your\s+)?password|verify\s+your\s+account|account\s+will\s+be\s+(locked|suspended)|"
    r"password\s+expires)",
    re.IGNORECASE,
)


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _lookalike(domain: str, org_domains: list[str]) -> bool:
    label = domain.split(".")[0].split("-")[0]
    for org in org_domains:
        org_label = org.split(".")[0]
        if domain == org or domain.endswith("." + org):
            return False
        if label != org_label and _levenshtein(label, org_label) <= 2:
            return True
    return False


class ThreatDetector:
    def __init__(self, org_domains: list[str], threshold: float = 0.5):
        self.org_domains = [d.lower() for d in org_domains]
        self.threshold = threshold

    def _scan(self, raw: str) -> list[tuple[str, float]]:
        hits: list[tuple[str, float]] = []
        text = normalise(raw)
        if INVISIBLE.search(raw):
            hits.append(("invisible_characters", 0.35))
        for sig in TEXT_SIGNALS:
            if sig.pattern.search(text):
                hits.append((sig.name, sig.weight))
        for comment in HTML_COMMENT.findall(text):
            if re.search(r"\b(ai|assistant|forward|send|ignore)\b", comment, re.IGNORECASE):
                hits.append(("instruction_in_html_comment", 0.6))
        for blob in BASE64_BLOB.findall(text):
            try:
                decoded = base64.b64decode(blob, validate=True).decode("utf-8")
            except (binascii.Error, UnicodeDecodeError, ValueError):
                continue
            nested = [n for n, _ in self._scan(decoded)]
            if nested or re.search(r"\b(send|forward|delete)\b", decoded, re.IGNORECASE):
                hits.append(("encoded_instruction", 0.6))
        return hits

    def assess(self, thread: EmailThread) -> GuardResult:
        signals: dict[str, float] = {}
        phishing = False
        for msg in thread.messages:
            if msg.direction != "inbound":
                continue
            for name, w in self._scan(f"{msg.subject}\n{msg.body}"):
                signals[name] = max(signals.get(name, 0.0), w)
            domains = {urlparse(u).hostname or "" for u in URL.findall(msg.body)}
            sender_lookalike = _lookalike(msg.sender.domain, self.org_domains)
            link_lookalike = any(_lookalike(d, self.org_domains) for d in domains if d)
            if (sender_lookalike or link_lookalike) and PHISHING_WORDS.search(msg.body):
                signals["lookalike_domain_credential_lure"] = 0.8
                phishing = True
            elif sender_lookalike or link_lookalike:
                signals["lookalike_domain"] = 0.3
        risk = 1.0
        for w in signals.values():
            risk *= 1.0 - w
        risk = round(1.0 - risk, 3)
        is_threat = risk >= self.threshold
        threat_type = None
        if is_threat:
            weak = {"lookalike_domain_credential_lure", "lookalike_domain", "exfiltration"}
            injection_like = set(signals) - weak
            threat_type = "phishing" if phishing and not injection_like else "prompt_injection"
        return GuardResult(
            risk_score=risk, signals=sorted(signals), is_threat=is_threat, threat_type=threat_type
        )
