"""egress redline consumer: redacts secret/PII slices at the moment they are materialized downstream.

The rule set is the single source pointed to by config (redaction_rules.toml, shared by CAP egress +
LED); this module only consumes rules, it does not own them. Each match is replaced wholesale with
[REDACTED:<category>:<name>], erring on the side of over-redaction for safety. A rule may set
`validate = "<name>"` to gate its regex matches through a named checker here (e.g. luhn for
credit_card), keeping sequences that fail the check in the stream as evidence.

Rules are collected against the original text and applied in one pass, so no rule ever sees another
rule's placeholder: the placeholder is the only record of what was removed, and a sequential sweep
let a later rule (api_key_assignment) match the `secret:<name>` inside an earlier rule's placeholder
and rewrite it as a nested, generic one. Placeholders already in the input are held the same way:
redacted text is fed back in (capture's evidence is redacted again when it lands).
"""
import functools
import re
import tomllib
from pathlib import Path

from autoharness import config


def _passes_luhn(text):
    digits = re.sub(r"\D", "", text)
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_VALIDATORS = {"luhn": _passes_luhn}


@functools.lru_cache(maxsize=4)
def _rules(rules_path):
    path = Path(rules_path) if rules_path else config.REDACTION_RULES
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    compiled = []
    for category in ("secret", "pii"):
        for rule in data.get(category, []):
            validate = rule.get("validate")
            if validate is not None and validate not in _VALIDATORS:
                raise ValueError(
                    f"unknown validator {validate!r} on rule {category}.{rule['name']}"
                )
            compiled.append(
                (category, rule["name"], re.compile(rule["pattern"]), _VALIDATORS.get(validate))
            )
    return compiled


def _token(category, name):
    return f"[REDACTED:{category}:{name}]"


@functools.lru_cache(maxsize=4)
def _placeholder(key):
    # only this rule set's own tokens: a made-up `[REDACTED:secret:<a real key>]` is still raw text
    return re.compile("|".join(re.escape(_token(category, name)) for category, name, _, _ in _rules(key)))


def _held(text, key):
    return [match.span() for match in _placeholder(key).finditer(text)]


def _overlaps(spans, start, end):
    return any(start < span[1] and span[0] < end for span in spans)


def redact(text, rules_path=None):
    spans = []
    key = str(rules_path) if rules_path else str(config.REDACTION_RULES)
    held = _held(text, key)
    for category, name, rx, validator in _rules(key):
        for match in rx.finditer(text):
            if validator is not None and not validator(match.group(0)):
                continue
            if _overlaps(held, match.start(), match.end()) or _overlaps(spans, match.start(), match.end()):
                continue
            spans.append((match.start(), match.end(), _token(category, name)))
    spans.sort(key=lambda span: span[0])

    out = []
    cursor = 0
    for start, end, token in spans:
        out.append(text[cursor:start])
        out.append(token)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def secret_hits(text, rules_path=None):
    """Names of the secret rules *text* matches — a gate, not a rewrite (PII stays redact-only)."""
    key = str(rules_path) if rules_path else str(config.REDACTION_RULES)
    held = _held(text, key)
    return [
        name
        for category, name, rx, validator in _rules(key)
        if category == "secret"
        and any(
            (validator is None or validator(match.group(0))) and not _overlaps(held, match.start(), match.end())
            for match in rx.finditer(text)
        )
    ]
