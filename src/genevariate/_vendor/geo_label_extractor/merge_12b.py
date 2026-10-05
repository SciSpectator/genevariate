"""Finalize: concat phase2 shard outputs -> apply sex-grounding and Age
grounding + formatting -> write merged corpus."""

import json, sys, os
import os as _os

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import sex_ground
import repair_age
import age_format

AGE_TEXT_KEYS = (
    "characteristics",
    "title",
    "source",
    "description",
    "treatment_protocol",
    "source_name",
    "gsm_title",
)


TMP = os.environ.get("GEO_TMP", "/dev/shm/geo_tmp")
os.makedirs(TMP, exist_ok=True)

rows = []
for i in range(4):
    d = json.load(open(f"{TMP}/p2out{i}.json"))
    part = d["samples"] if isinstance(d, dict) and "samples" in d else d
    rows.extend(part)
    print(f"shard{i}: {len(part)} samples", flush=True)
print(f"merged total: {len(rows)} samples", flush=True)

sex_blanked = 0
age_grounded = 0
age_reformatted = 0
for s in rows:
    try:
        if sex_ground.apply_sex_grounding(s):
            sex_blanked += 1
    except Exception as e:
        print(f"sex_ground error {s.get('gsm')}: {e!r}", flush=True)

    try:
        txt = " ".join(str(s.get(k, "")) for k in AGE_TEXT_KEYS)
        hit = False
        for ph in ("phase1b", "phase2"):
            d = s.get(ph)
            if isinstance(d, dict) and "Age" in d:
                new, _why = repair_age.repair_age(d["Age"], txt)
                if new != d["Age"]:
                    d["Age"] = new
                    hit = True
        if hit:
            age_grounded += 1
        if age_format.apply_age_format(s):
            age_reformatted += 1
    except Exception as e:
        print(f"age error {s.get('gsm')}: {e!r}", flush=True)

print(f"sex-grounding blanked: {sex_blanked}", flush=True)
print(f"age grounding changed: {age_grounded}", flush=True)
print(f"age key-prefix reformatted: {age_reformatted}", flush=True)

json.dump(
    {"samples": rows, "n_samples": len(rows)},
    open(f"{TMP}/merged_12b.json", "w"),
    default=str,
)
print(f"wrote {TMP}/merged_12b.json", flush=True)
