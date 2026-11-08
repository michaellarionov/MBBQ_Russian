"""95% confidence intervals for the directional shift (cluster bootstrap over templates).

Score per ambiguous answer: +1 = names the man, -1 = names the woman, 0 = unknown.
For each comparison (a vs b), pair the two versions on the same item and prompt,
take d = score(a) - score(b), and report mean(d) x 100 = shift in points toward
the man (negative = toward the woman). The 95% CI comes from resampling the 17
templates with replacement (all items of a template stay together), 10,000 draws,
2.5th / 97.5th percentiles.

Usage:
    python compute_ci95.py --root /path/to/MBBQ_Russian [--draws 10000] [--seed 0] [--out ci95.csv]

Needs pandas and numpy. Uses the repo's own loading and classification
(my_research/scorer/fem_analysis.py, score_both_signs.py) and regenerates the
Russian data with my_research/templates/generate_ru.py.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# The scorer modules import the generation stack; stub it so this runs without
# torch/transformers installed. Harmless if they are installed (e.g. in Colab).
for _m in ["torch", "transformers", "transformers.pipelines",
           "transformers.pipelines.pt_utils", "datasets", "accelerate", "tqdm"]:
    if _m not in sys.modules:
        try:
            __import__(_m)
        except ImportError:
            sys.modules[_m] = MagicMock()
if isinstance(sys.modules.get("torch"), MagicMock):
    sys.modules["torch"].Tensor = type("Tensor", (), {})

import numpy as np
import pandas as pd

COMPARISONS = [("masc", "main"), ("fem", "masc"), ("masc_rerun", "masc")]
LABEL = {"main": "neutral", "masc": "masc", "fem": "fem", "masc_rerun": "rerun"}


def score(pick):
    return (pick == "m").astype(int) - (pick == "f").astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to the MBBQ_Russian repo")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ci95.csv")
    args = ap.parse_args()

    root = Path(args.root)
    sys.path.insert(0, str(root / "my_research/scorer"))
    import fem_analysis as F
    from score_both_signs import Outputs

    rng = np.random.default_rng(args.seed)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        g = subprocess.run([sys.executable, str(root / "my_research/templates/generate_ru.py"),
                            "--out", str(data)], capture_output=True, text=True)
        if g.returncode:
            sys.exit(g.stdout + g.stderr)
        out = Outputs(root / "my_research/results", root / "data", data)

        for m in F.MODELS:
            sets = {}
            for s in ["main", "masc", "fem", "masc_rerun"]:
                raw = out.get(m, "ru", s)
                raw = raw[raw.subset == "Gender_identity"]
                sets[s] = F.classify(raw, F.option_genders(data / F.DATA_FILE[s]))
            # restrict every set to the 144 items shared with fem
            items = set(zip(sets["fem"].subset, sets["fem"].example_id))
            sets = {s: d[[k in items for k in zip(d.subset, d.example_id)]] for s, d in sets.items()}

            for a, b in COMPARISONS:
                mm = sets[a].merge(sets[b], on=F.KEY, suffixes=("_a", "_b"))
                x = mm[mm.pick_a.notna() & mm.pick_b.notna()
                       & (mm.context_condition_a == "ambig")].copy()
                x["d"] = score(x.pick_a) - score(x.pick_b)

                t = x.groupby("q_id_a").d.agg(["sum", "count"])
                ids = t.index.values
                bs = np.empty(args.draws)
                for i in range(args.draws):
                    s = t.loc[rng.choice(ids, len(ids))]
                    bs[i] = s["sum"].sum() / s["count"].sum()
                est = x.d.mean() * 100
                lo, hi = np.percentile(bs, [2.5, 97.5]) * 100
                rows.append(dict(model=m, comparison=f"{LABEL[a]} vs {LABEL[b]}",
                                 n_pairs=len(x), n_templates=len(ids),
                                 estimate_pts=round(est, 1),
                                 ci95_low=round(lo, 1), ci95_high=round(hi, 1)))
                print(f"{m:8s} {LABEL[a]:6s} vs {LABEL[b]:7s} "
                      f"{est:+5.1f} pts  95% CI [{lo:+5.1f}, {hi:+5.1f}]  "
                      f"({len(x)} pairs, {len(ids)} templates)")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nSaved {args.out}  (positive = toward the man, negative = toward the woman)")


if __name__ == "__main__":
    main()
