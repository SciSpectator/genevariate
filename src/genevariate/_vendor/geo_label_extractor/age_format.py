"""Age output formatting: drop the echoed characteristics key.

Recognise a leading "<key>:" where the key is either an age descriptor or a
time unit, drop it, and re-attach the unit as a suffix so no information is
lost. Anything whose key is not age-ish is left exactly as-is.

Runs AFTER repair_age (which parses the "unit:" form to decide plausibility).
"""

import re

NS = "Not Specified"

_UNITS = {
    "year": "years",
    "years": "years",
    "yr": "years",
    "yrs": "years",
    "month": "months",
    "months": "months",
    "mo": "months",
    "mos": "months",
    "week": "weeks",
    "weeks": "weeks",
    "wk": "weeks",
    "wks": "weeks",
    "day": "days",
    "days": "days",
    "hour": "hours",
    "hours": "hours",
    "hr": "hours",
    "hrs": "hours",
}

_KEYED = re.compile(
    r"^\s*(?P<key>[A-Za-z][A-Za-z_ /-]{0,24}?)\s*"
    r"(?:\((?P<paren>[^)]*)\))?\s*[:=]\s*(?P<val>.+?)\s*$"
)


_NUMERIC = re.compile(r"^[~<>=]*\s*\d+(?:\.\d+)?\s*(?:[-–—+]\s*\d+(?:\.\d+)?)?\s*$")
_HAS_UNIT = re.compile(
    r"\b(?:years?|yrs?|months?|mos?|weeks?|wks?|days?|hours?|hrs?|" r"y/?o|gw|pnd?)\b",
    re.I,
)


def normalize_age(value):
    """'years: 62' -> '62 years'; 'age: adult' -> 'adult'; leaves non-age keys alone."""
    if value is None:
        return value
    s = str(value).strip()
    if not s or s.lower() == NS.lower():
        return value
    m = _KEYED.match(s)
    if not m:
        return value

    key = re.sub(r"[\s_/-]+", " ", m.group("key").strip().lower())
    paren = (m.group("paren") or "").strip().lower()
    val = m.group("val").strip()
    if not val:
        return NS

    unit = _UNITS.get(paren)
    if key in _UNITS:
        unit = unit or _UNITS[key]
    elif "age" not in key.split():
        return value

    if _HAS_UNIT.search(val):
        return val
    if _NUMERIC.match(val):

        return f"{val} {unit or 'years'}"
    return f"{val} {unit}" if unit else val


def apply_age_format(sample):
    """Normalise Age across phase1b + phase2 in place. True if anything changed."""
    changed = False
    for phase in ("phase1b", "phase2"):
        d = sample.get(phase)
        if isinstance(d, dict) and "Age" in d:
            new = normalize_age(d["Age"])
            if new != d["Age"]:
                d["Age"] = new
                changed = True
    return changed


if __name__ == "__main__":
    tests = [
        ("years: 62", "62 years"),
        ("age: 62", "62 years"),
        ("age: adult", "adult"),
        ("weeks: 8", "8 weeks"),
        ("months: 3", "3 months"),
        ("days: 14", "14 days"),
        ("age (years): 56", "56 years"),
        ("age: 8 weeks", "8 weeks"),
        ("yrs: 45", "45 years"),
        ("age: 56-60", "56-60 years"),
        ("donor age: 31", "31 years"),
        ("age: ~70", "~70 years"),
        ("62 years", "62 years"),
        ("adult", "adult"),
        ("Not Specified", "Not Specified"),
        ("cell line: HeLa", "cell line: HeLa"),
        ("passage: 12", "passage: 12"),
    ]
    bad = 0
    for src, exp in tests:
        got = normalize_age(src)
        if got != exp:
            bad += 1
        verdict = "OK " if got == exp else "FAIL"
        print(f"[{verdict}] {src!r:22} -> {got!r:18} (exp {exp!r})")
    print("ALL PASS" if not bad else f"{bad} FAILED")
