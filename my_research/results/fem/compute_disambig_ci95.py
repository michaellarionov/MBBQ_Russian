"""95% CIs for the change in disambiguated accuracy (template-cluster bootstrap).

For every row of Table 2 (masc vs neutral, fem vs masc, rerun vs masc, per model),
pair the two versions on the same disambiguated item and prompt, split by the
gold person's gender, and take d = correct(a) - correct(b). The estimate is
mean(d) x 100 = change in accuracy points under the first set. The 95% CI comes
from resampling the 17 templates with replacement (all items of a template stay
together), 10,000 draws, 2.5th / 97.5th percentiles.

Uses seed + 1, the same stream compute_ci95.py uses for these outcomes, so the
numbers match its disambig_acc_* rows exactly.

Usage:
    python compute_disambig_ci95.py --root /path/to/MBBQ_Russian [--draws 10000] [--seed 0] [--out disambig_ci95.csv]

Needs pandas and numpy.
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



def bootstrap(x, rng, draws):
    """Mean of x.d x 100 and its 95% CI, resampling templates with replacement."""
    t = x.groupby("q_id_a").d.agg(["sum", "count"])
    ids = t.index.values
    bs = np.empty(draws)
    for i in range(draws):
        s = t.loc[rng.choice(ids, len(ids))]
        bs[i] = s["sum"].sum() / s["count"].sum()
    lo, hi = np.percentile(bs, [2.5, 97.5]) * 100
    return x.d.mean() * 100, lo, hi, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to the MBBQ_Russian repo")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="disambig_ci95.csv")
    args = ap.parse_args()

    root = Path(args.root)
    sys.path.insert(0, str(root / "my_research/scorer"))
    import fem_analysis as F
    from score_both_signs import Outputs

    rng = np.random.default_rng(args.seed + 1)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        g = subprocess.run([sys.executable, str(root / "my_research/templates/generate_ru.py"),
                            "--out", str(data)], capture_output=True, text=True)
        if g.returncode:
            sys.exit(g.stdout + g.stderr)
        out = Outputs(root / "my_research/results", root / "data", data)

        print(f"{'model':8s} {'comparison':17s} {'gold woman':>24s} {'gold man':>24s}")
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
                dis = mm[mm.pick_a.notna() & mm.pick_b.notna()
                         & (mm.context_condition_a == "disambig")]
                row = dict(model=m, comparison=f"{LABEL[a]} vs {LABEL[b]}")
                cells = []
                for g, name in [("f", "gold_woman"), ("m", "gold_man")]:
                    y = dis[dis.gold_gender_a == g].copy()
                    y["d"] = ((y.pick_a == y.gold_gender_a).astype(int)
                              - (y.pick_b == y.gold_gender_a).astype(int))
                    est, lo, hi, n_t = bootstrap(y, rng, args.draws)
                    row.update({f"{name}_n_pairs": len(y),
                                f"{name}_estimate_pts": round(est, 1),
                                f"{name}_ci95_low": round(lo, 1),
                                f"{name}_ci95_high": round(hi, 1)})
                    cells.append(f"{est:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]")
                row["n_templates"] = n_t
                rows.append(row)
                print(f"{m:8s} {row['comparison']:17s} {cells[0]:>24s} {cells[1]:>24s}")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nSaved {args.out}  (accuracy points; positive = higher accuracy under the first set)")


if __name__ == "__main__":
    main()
