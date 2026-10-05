"""Deterministic, provenance-aware age repair for the age output — backstops the
2B model's tendency to emit experimental/clinical TIME as age. Blanks age values
that are actually a duration / timepoint (survival, relapse, follow-up, training,
days/weeks post-X, culture, etc.) with no genuine age cue, plus computed/epigenetic
ages and impossible years. Preserves real ages: explicit age fields, pediatric
day/week ages, and qualitative life stages. Scalable — general cue categories only,
no hardcoded study names.

Usage: repair_age.py <age_output_dir> [--write gsm_age.json]
Scans ck_age_*.jsonl, prints impact + examples; with --write emits {gsm: age} for
the merge step to use as the authoritative Age."""

import json, glob, re, os, sys, collections, argparse

NS = "Not Specified"
AGE_CUE = re.compile(
    r"\bage\b\s*(?:\([^)]*\))?\s*[:=]|donor.?age|patient.?age|maternal.?age|"
    r"gestational|postnatal|post-?conception|of gestation|\bpnd\s*\d|"
    r"\d[\d.\s-]*\s*(?:day|week|month|year|yr|wk|mo)s?\s*old|at (?:birth|diagnosis|death)|"
    r"newborn|neonat|fetal|foetal|premature|carnegie",
    re.I,
)
TIME_CTX = re.compile(
    r"\btime\b|timepoint|\bsession\b|\bsurv|relaps|recurren|follow[-\s]?up|"
    r"disease[-\s]?free|progression[-\s]?free|(?:months?|weeks?|days?|time)\s+to\b|latency|"
    r"post[-\s]?(?:infection|treatment|stimulation|transfection|challenge|surgery|induction|exposure|op)|"
    r"\bdays?\s+post\b|\bweeks?\s+post\b|\bdpi\b|\bhpi\b|passage|expansion|differentiat|"
    r"knock-?down|\bafter\s+\d|treat(?:ment|ed)?|stimulat|infect|exposure|incubat|induct|"
    r"engraft|training|intervention|regimen|harvest|\bcultured?\b|challenge|\bdose\b|"
    r"collected?\s+(?:at|on|after)|elapsed|duration",
    re.I,
)
CELLLINE = re.compile(r"cell line\s*:", re.I)
LIFE_STAGE = re.compile(
    r"newborn|neonat|infant|juvenile|adult|elderly|fetal|foetal|embryo|child|"
    r"adolescen|geriatric|pediatric|prenatal|larval|pupa",
    re.I,
)
COMPUTED_AGE = re.compile(
    r"dnam\s*age|methylation\s*age|predicted\s*age|epigenetic\s*age|biological\s*age|"
    r"estimated\s*age|age\s*acceleration|horvath|hannum|transcriptomic\s*age|clock\s*age",
    re.I,
)
CHRONO_AGE_NUM = re.compile(r"\bage\b\s*(?:\([^)]*\))?\s*[:=]\s*~?\s*\d", re.I)
_UNIT_MAP = {
    "day": "days",
    "week": "weeks",
    "month": "months",
    "hour": "hours",
    "yr": "years",
    "year": "years",
}
_NONYEAR = ("days", "day", "weeks", "week", "months", "month", "mo", "hours", "hour")


def repair_age(label, text):
    if not label or str(label).strip().lower() in ("", NS.lower()):
        return NS, "already-blank"
    s = str(label).strip()
    low = s.lower()
    nums = re.findall(r"\d+\.?\d*", s)
    if not nums:
        return (
            (label, "keep-lifestage")
            if LIFE_STAGE.search(s)
            else (NS, "no-number-junk")
        )
    m = re.match(r"([a-z]+)\s*:", low)
    unit = m.group(1) if m else None
    if unit not in _NONYEAR + ("years", "year", "yr"):
        vm = re.search(r"\d\s*(day|week|month|hour|yr|year)s?\b", low)
        if vm:
            unit = _UNIT_MAP[vm.group(1)]

    if unit in (None, "age"):
        unit = "years"
    if s.count(";") >= 1 or len(nums) > 2:
        return NS, "multi-value"
    if COMPUTED_AGE.search(text):
        twx = COMPUTED_AGE.sub(" ", text)
        many_dec = any("." in n and len(n.split(".")[1]) >= 2 for n in nums)
        if not CHRONO_AGE_NUM.search(twx) or many_dec:
            return NS, "blank-computed-age"
    try:
        if unit in ("years", "year", "yr") and any(float(n) > 120 for n in nums):
            return NS, "years>120"
    except ValueError:
        pass

    if TIME_CTX.search(text) and not AGE_CUE.search(text):
        return NS, "blank-time-ctx"

    grounded = bool(
        AGE_CUE.search(text)
        or LIFE_STAGE.search(text)
        or re.search(
            r"\d\s*(?:years?|yrs?|y/?o|months?|mos?|weeks?|wks?|days?|hours?|hrs?)\b",
            text,
            re.I,
        )
        or re.search(
            r"\bGW\s*\d|\bE\d+\.\d|gestat|postnatal|\bPND?\s*\d|blastocyst|larval",
            text,
            re.I,
        )
    )
    if not grounded:
        return NS, "blank-ungrounded-number"
    if unit in _NONYEAR:
        if AGE_CUE.search(text):
            return label, "keep-explicit-age"
        if CELLLINE.search(text):
            return NS, "blank-cellline"
        return label, "keep-default"
    return label, "keep"


def auth_age(it):
    for ph in ("phase2", "phase1b", "phase1"):
        v = it.get(ph)
        if isinstance(v, dict) and v.get("Age") not in (None, ""):
            return v["Age"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("age_dir")
    ap.add_argument("--write")
    a = ap.parse_args()
    reason = collections.Counter()
    blanked = []
    out = {}
    n = 0
    for fn in sorted(glob.glob(f"{a.age_dir}/ck_age_*.jsonl")):
        for line in open(fn, encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            for it in obj.get("samples", []):
                if not isinstance(it, dict) or "gsm" not in it:
                    continue
                n += 1
                l = auth_age(it)
                txt = " ".join(
                    str(it.get(k, ""))
                    for k in (
                        "characteristics",
                        "title",
                        "source",
                        "description",
                        "treatment_protocol",
                        "source_name",
                        "gsm_title",
                    )
                )
                new, why = repair_age(l, txt)
                reason[why] += 1
                out[it["gsm"]] = new
                if (
                    why.startswith("blank")
                    and l
                    and str(l).lower() != NS.lower()
                    and len(blanked) < 10
                ):
                    blanked.append((why, str(l), txt[:80].replace(chr(9), " ")))
    changed = sum(
        reason[k]
        for k in reason
        if k.startswith("blank") or k in ("multi-value", "years>120")
    )
    print(
        f"age samples: {n} | repaired -> Not Specified: {changed} ({100 * changed / max(n ,1):.1f}%)"
    )
    for k, c in reason.most_common():
        print(f"   {c :7d}  {k}")
    for w, l, ch in blanked:
        print(f"   [{w}] {l!r} <= {ch}")
    if a.write:
        json.dump(out, open(a.write, "w"))
        print(f"wrote {len(out)} gsm->age to {a.write}")


if __name__ == "__main__" and "--selftest" in sys.argv:
    cases = [
        ("age: 14 months", "months to relapse: 14 T-ALL", NS),
        ("age: 3 months", "surv(months): 3 status", NS),
        ("age: 4678 days", "length of follow-up (days): 4678", NS),
        (
            "age: 8 weeks",
            "muscle biopsy after an eight weeks endurance training intervention",
            NS,
        ),
        ("years: 55", "age: 55; tissue: blood", "years: 55"),
        ("days: 66", "age of biopsy: 66 days; biliary atresia", "days: 66"),
        ("years: 30", "age (yrs): 30; healthy", "years: 30"),
    ]
    ok = True
    for lab, txt, want in cases:
        got, why = repair_age(lab, txt)
        blanked = str(got).lower() == NS.lower()
        exp_blank = str(want).lower() == NS.lower()
        good = blanked == exp_blank
        ok = ok and good
        verdict = "PASS" if good else "FAIL"
        print(f"  {verdict}  {lab!r:20} -> {got!r:16} ({why})")
    print("SELFTEST:", "ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
