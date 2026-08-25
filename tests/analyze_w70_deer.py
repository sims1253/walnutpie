#!/usr/bin/env python3
"""W-70 DEER/Picard feasibility: summarize driver RESULT lines into tables."""
import glob
import re
import sys

pat = re.compile(
    r"RESULT model=(\S+) median_rounds=(\d+) p90_rounds=(\d+) "
    r"frac_le3=([\d.e+-]+) strict=(\d+) relaxed_only=(\d+) fail=(\d+) "
    r"seq_evals=(\d+) picard_evals=(\d+) overhead=([\d.e+-]+)")
warm = re.compile(r"# warmup done: eps=(\S+)")
errln = re.compile(r"# err_abs med=(\S+) p90=(\S+) max=(\S+)")

def label(path):
    name = path.split("/")[-1].replace(".log", "")
    arm = "diag-metric" if name.endswith("_met") else (
        "identity-metric" if name.endswith("_idmet") else "?")
    return name.replace("_idmet", "").replace("_met", ""), arm

rows = []
for f in sorted(sys.argv[1] and glob.glob(sys.argv[1]) or []):
    txt = open(f, errors="ignore").read()
    m = pat.search(txt)
    if not m:
        continue
    model, arm = label(f)
    w = warm.search(txt)
    e = errln.search(txt)
    n = 200
    strict = int(m.group(5))
    rows.append(dict(model=model, arm=arm, eps=w.group(1) if w else "?",
                     med=m.group(2), p90=m.group(3), frac=m.group(4),
                     strict=strict, conv=strict + int(m.group(6)),
                     fail=int(m.group(7)), ovr=float(m.group(10)),
                     errmed=e.group(1) if e else "?"))

print(f"{'model':<24}{'arm':<17}{'eps(frozen)':<12}{'med':>4}{'p90':>4}"
      f"{'conv':>6}{'fail':>6}{'frac<=3':>9}{'overhead':>10}"
      f"{'final_err(med)':>16}")
for r in rows:
    print(f"{r['model']:<24}{r['arm']:<17}{r['eps']:<12.11}{r['med']:>4}"
          f"{r['p90']:>4}{r['conv']:>6}{r['fail']:>6}{r['frac']:>9}"
          f"{r['ovr']:>10.2f}{r['errmed']:>16}")
