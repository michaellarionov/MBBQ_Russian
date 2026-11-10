"""Checks reviewers will ask for on the fem/masc study (Gender_identity, 17 templates).

  python3 my_research/scorer/fem_reviewer_checks.py [--draws 10000] [--seed 2]
      [--out my_research/results/fem]

Uses the same loading, re-detection, item restriction (the 144 `fem` items x 5
prompts) and woman/man/unknown classes as fem_analysis.py.

1. Wrong-answer breakdown, disambiguated contexts. Per set and gold gender:
   shares of all rows that are correct / the other person / «unknown» /
   undetected. Then paired decompositions (rows detected in both sets, as in
   disambig_ci95.csv): the change in accuracy splits exactly into
   -(change in «other person») - (change in «unknown»). Each part gets a
   template-cluster bootstrap 95% CI (17 templates resampled, percentile).
2. Template-level primary tests. H1 («woman», fem vs. masc) and H2 («man»,
   fem vs. masc; a < b supports it) as two-sided Wilcoxon signed-rank tests
   over the 17 per-template mean differences in ambiguous contexts (zero
   differences dropped, as in fem_analysis.py). p is exact: all 2^n sign
   flips of the signed (mid)ranks, so ties are handled. Holm within the same
   families as the primary sign tests.
3. What the masc-vs-neutral bootstrap rules out: read from ci95.csv
   (detection/compute_ci95.py), not recomputed here.

Writes REVIEWER_CHECKS.md, wrong_answers.csv and template_tests.csv to --out.
"""
import argparse
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fem_analysis as F  # noqa: E402
from score_both_signs import Outputs  # noqa: E402

SETS = ["main", "masc", "fem", "masc_rerun"]
LABEL = {"main": "neutral", "masc": "masc", "fem": "fem", "masc_rerun": "rerun"}
NAME = {"f": "woman", "m": "man", "u": "unknown"}
OTHER = {"f": "m", "m": "f"}
TEMPLATE = ["subset", "q_id_a"]
# (a, b, gold gender, why)
DECOMPOSE = [("fem", "masc", "m", "asked: where fem's lost gold-man accuracy goes"),
             ("fem", "masc", "f", "mirror: where fem's gained gold-woman accuracy comes from"),
             ("masc", "main", "f", "masculine default: gold-woman items, masc vs. neutral"),
             ("masc", "main", "m", "masculine default: gold-man items, masc vs. neutral"),
             ("masc_rerun", "masc", "m", "noise floor, gold man"),
             ("masc_rerun", "masc", "f", "noise floor, gold woman")]


def load(results):
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        gen = subprocess.run([sys.executable, str(ROOT / "my_research/templates/generate_ru.py"),
                              "--out", str(data)], capture_output=True, text=True)
        if gen.returncode:
            sys.exit(f"generate_ru.py failed:\n{gen.stdout}{gen.stderr}")
        out = Outputs(results, ROOT / "data", data)
        sets = {}
        models = [m for m in F.MODELS if all(out.get(m, "ru", s) is not None for s in SETS)]
        for m in models:
            have = {}
            for s in SETS:
                raw = F.settle_lone_v(m, out.get(m, "ru", s))
                raw = raw[raw["subset"] == "Gender_identity"]
                have[s] = F.classify(raw, F.option_genders(data / F.DATA_FILE[s]))
            items = set(zip(have["fem"]["subset"], have["fem"]["example_id"]))
            for s, d in have.items():
                sets[(m, s)] = d[[k in items for k in zip(d["subset"], d["example_id"])]]
    return sets, out, models


def boot_ci(x, cols, rng, draws):
    """Template-cluster percentile 95% CIs for the means of x[cols] (x 100)."""
    t = x.groupby(TEMPLATE)[cols].agg(["sum", "count"])
    n = len(t)
    sums = np.stack([t[(c, "sum")].to_numpy(float) for c in cols])
    cnt = t[(cols[0], "count")].to_numpy(float)
    idx = rng.integers(0, n, size=(draws, n))
    bs = sums[:, idx].sum(axis=2) / cnt[idx].sum(axis=1)  # (cols, draws)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=1) * 100
    return {c: (lo[i], hi[i]) for i, c in enumerate(cols)}, n


def exact_wilcoxon(diffs):
    """Two-sided exact Wilcoxon signed-rank p over all sign flips (zeros dropped)."""
    d = np.asarray([v for v in diffs if v != 0], float)
    if not len(d):
        return 1.0, 0
    r = rankdata(np.abs(d))
    w = r[d > 0].sum()
    centre = r.sum() / 2
    flips = np.array(list(itertools.product([0, 1], repeat=len(d))), float)
    w_all = flips @ r
    p = np.mean(np.abs(w_all - centre) >= abs(w - centre) - 1e-9)
    return float(p), len(d)


def fmt_p(p):
    return F.fmt_p(p)


def pts(v):
    return f"{v:+.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=ROOT / "my_research" / "results")
    ap.add_argument("--out", default=ROOT / "my_research" / "results" / "fem")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    out_dir = Path(args.out)
    rng = np.random.default_rng(args.seed)
    # Supplementary models (Aya) draw from their own stream, so adding them
    # leaves the main models' bootstrap draws exactly as before.
    rng_supp = np.random.default_rng(args.seed + 1000)
    sets, outputs, models = load(args.results)

    L = ["# Reviewer checks: wrong answers, template-level tests, bounds", "",
         "Generated by `my_research/scorer/fem_reviewer_checks.py`. Same sets, re-detection",
         "and 144 items × 5 prompts as `FEM_RESULTS.md`. Points = percentage points.",
         f"Bootstrap: {args.draws:,} template-cluster draws (17 templates), seed {args.seed}",
         f"(supplementary models: seed {args.seed + 1000}).", ""]

    # ---- 1. wrong-answer breakdown -------------------------------------------
    L += ["## 1. Where disambiguated answers go", "",
          "### Shares of all disambiguated rows, by gold gender", "",
          "Percent of all rows (undetected included), so each row sums to 100.", "",
          "| Model | Set | Gold | Rows | Correct | Other person | Unknown | Undetected |",
          "|---|---|---|---|---|---|---|---|"]
    wrong_rows = []
    for m in models:
        for s in SETS:
            d = sets[(m, s)]
            d = d[d["context_condition"] == "disambig"]
            for g in ("m", "f"):
                x = d[d["gold_gender"] == g]
                n = len(x)
                c = {"correct": (x["pick"] == g).sum(), "other": (x["pick"] == OTHER[g]).sum(),
                     "unknown": (x["pick"] == "u").sum(), "undetected": x["pick"].isna().sum()}
                assert sum(c.values()) == n, (m, s, g)
                wrong_rows.append(dict(model=m, set=LABEL[s], gold=NAME[g], rows=n,
                                       **{f"{k}_pct": 100 * v / n for k, v in c.items()}))
                L.append(f"| {m} | {LABEL[s]} | {NAME[g]} | {n} | "
                         + " | ".join(f"{100 * v / n:.1f}" for v in c.values()) + " |")
    L += ["", "### Paired changes, split into «other person» and «unknown»", "",
          "Rows detected in both sets. Δ accuracy = −(Δ other person) − (Δ unknown),",
          "exactly. «Share to other» = Δ other / (Δ other + Δ unknown) when accuracy",
          "falls and both parts rise; «all to X» if the other part fell; blank if accuracy",
          "rises or doesn't move. Transitions: rows correct under b",
          "and wrong under a (→ other / → unknown), and the reverse.", "",
          "| Model | a vs. b | Gold | Pairs | Δ acc [95% CI] | Δ other [95% CI] | Δ unknown [95% CI] "
          "| Share to other | Correct→other / →unknown | Other→correct / unknown→correct |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a, b, g, why in DECOMPOSE:
        for m in models:
            mm = sets[(m, a)].merge(sets[(m, b)], on=F.KEY, suffixes=("_a", "_b"))
            x = mm[mm["pick_a"].notna() & mm["pick_b"].notna()
                   & (mm["context_condition_a"] == "disambig") & (mm["gold_gender_a"] == g)].copy()
            o = OTHER[g]
            x["acc"] = (x["pick_a"] == g).astype(int) - (x["pick_b"] == g).astype(int)
            x["oth"] = (x["pick_a"] == o).astype(int) - (x["pick_b"] == o).astype(int)
            x["unk"] = (x["pick_a"] == "u").astype(int) - (x["pick_b"] == "u").astype(int)
            assert (x["acc"] + x["oth"] + x["unk"] == 0).all()
            ci, n_t = boot_ci(x, ["acc", "oth", "unk"],
                              rng_supp if m in F.SUPPLEMENTARY else rng, args.draws)
            est = {k: 100 * x[k].mean() for k in ("acc", "oth", "unk")}
            if est["acc"] >= 0:
                share = None  # accuracy rose or didn't move
            elif est["oth"] >= 0 and est["unk"] >= 0:
                share = f"{est['oth'] / -est['acc']:.0%}"
            else:  # one part fell: the other carries more than the whole loss
                share = "all to other" if est["oth"] > 0 else "all to unknown"
            was_ok, now_ok = x["pick_b"] == g, x["pick_a"] == g
            t = {"correct_to_other": int((was_ok & (x["pick_a"] == o)).sum()),
                 "correct_to_unknown": int((was_ok & (x["pick_a"] == "u")).sum()),
                 "other_to_correct": int((now_ok & (x["pick_b"] == o)).sum()),
                 "unknown_to_correct": int((now_ok & (x["pick_b"] == "u")).sum())}
            wrong_rows.append(dict(model=m, set=f"{LABEL[a]} vs {LABEL[b]}", gold=NAME[g],
                                   rows=len(x), n_templates=n_t, purpose=why,
                                   **{f"d_{k}_pts": est[k] for k in est},
                                   **{f"d_{k}_ci95_{e}": ci[k][j] for k in est
                                      for j, e in enumerate(("low", "high"))},
                                   share_to_other=share, **t))
            cell = lambda k: f"{pts(est[k])} [{pts(ci[k][0])}, {pts(ci[k][1])}]"  # noqa: E731
            L.append(f"| {m} | {LABEL[a]} vs. {LABEL[b]} | {NAME[g]} | {len(x)} | {cell('acc')} | "
                     f"{cell('oth')} | {cell('unk')} | "
                     f"{share or ''} | "
                     f"{t['correct_to_other']} / {t['correct_to_unknown']} | "
                     f"{t['other_to_correct']} / {t['unknown_to_correct']} |")

    # ---- 2. template-level primary tests ---------------------------------------
    L += ["", "## 2. Template-level primary tests (fem vs. masc, ambiguous)", "",
          "Per template: mean over its paired rows of [answer is X under fem] − [under masc].",
          "Two-sided exact Wilcoxon signed-rank over templates with a nonzero mean (zeros",
          "dropped); Holm within the same families as the item-level tests. The item-level",
          "sign test (FEM_RESULTS.md) is repeated for comparison. «Mean» = unweighted mean",
          "of the 17 template differences, in points.", "",
          "| Family | Model | Hypothesis | Templates fem>masc / fem<masc / 0 | Mean (pts) | "
          "Wilcoxon p | Holm p | Item sign test p | Item Holm p |", "|---|---|---|---|---|---|---|---|---|"]
    tests = []
    for m in models:
        mm = sets[(m, "fem")].merge(sets[(m, "masc")], on=F.KEY, suffixes=("_a", "_b"))
        amb = mm[mm["pick_a"].notna() & mm["pick_b"].notna() & (mm["context_condition_a"] == "ambig")]
        for g, h in (("f", "H1"), ("m", "H2")):
            d = (amb["pick_a"] == g).astype(int) - (amb["pick_b"] == g).astype(int)
            tmpl = d.groupby([amb["subset"], amb["q_id_a"]]).mean()
            item = d.groupby([amb["subset"], amb["example_id"]]).sum()
            p, n_nz = exact_wilcoxon(tmpl.to_numpy())
            nz = tmpl[tmpl != 0]
            p_scipy = wilcoxon(nz).pvalue if len(nz) else 1.0
            _, _, p_item = F.sign_test(item)
            pred = (tmpl > 0).sum() > (tmpl < 0).sum() if g == "f" else (tmpl < 0).sum() > (tmpl > 0).sum()
            tests.append(dict(model=m, hypothesis=h, n_templates=len(tmpl),
                              templates_fem_gt=int((tmpl > 0).sum()),
                              templates_fem_lt=int((tmpl < 0).sum()),
                              templates_zero=int((tmpl == 0).sum()),
                              mean_template_diff_pts=100 * tmpl.mean(),
                              predicted_direction=bool(pred), wilcoxon_exact_p=p,
                              wilcoxon_scipy_p=p_scipy, item_sign_p=p_item))
    for fam in F.FAMILIES:
        fam_tests = [t for t in tests if t["model"] in fam]
        for t, q, qi in zip(fam_tests, F.holm([t["wilcoxon_exact_p"] for t in fam_tests]),
                            F.holm([t["item_sign_p"] for t in fam_tests])):
            t["wilcoxon_holm_p"], t["item_holm_p"], t["family"] = q, qi, " + ".join(fam)
            hyp = ("H1: more «woman» under fem" if t["hypothesis"] == "H1"
                   else "H2: more «man» under masc")
            L.append(f"| {t['family']} | {t['model']} | {hyp} | {t['templates_fem_gt']} / "
                     f"{t['templates_fem_lt']} / {t['templates_zero']} | "
                     f"{pts(t['mean_template_diff_pts'])} | {fmt_p(t['wilcoxon_exact_p'])} | "
                     f"{fmt_p(q)} | {fmt_p(t['item_sign_p'])} | {fmt_p(qi)} |")
    L += ["", "H2 counts: fem<masc templates support it (fewer «man» under fem).",
          "scipy's `wilcoxon` (the Wilcoxon column of FEM_RESULTS.md) falls back to the",
          "normal approximation when template differences tie; the exact p differs only",
          "for Zephyr H1 (exact 3.1e-04, scipy 0.0012; both in template_tests.csv).", ""]
    for m in models:
        mt = [t for t in tests if t["model"] == m]
        ok = lambda k: all(t["predicted_direction"] and t[k] < 0.05 for t in mt)  # noqa: E731
        L.append(f"- {m}: template level {'both hold' if ok('wilcoxon_holm_p') else 'not both'}"
                 f" (item level: {'both hold' if ok('item_holm_p') else 'not both'})")

    # ---- 3. bounds from ci95.csv ------------------------------------------------
    ci = pd.read_csv(out_dir / "ci95.csv")
    mv = ci[ci["comparison"] == "masc vs neutral"].set_index("model")
    L += ["", "## 3. What the masc-vs-neutral bootstrap rules out", "",
          "From `ci95.csv` (`detection/compute_ci95.py`, seed 0): directional score +1 man,",
          "−1 woman, 0 unknown, per ambiguous answer; masc minus neutral, in points;",
          "positive = toward the man. The upper bound is the largest masculine shift the",
          "95% CI still allows.", "",
          "| Model | Pairs | Estimate | 95% CI | Masculine shift larger than this excluded |",
          "|---|---|---|---|---|"]
    for m in [m for m in models if m in mv.index]:
        r = mv.loc[m]
        L.append(f"| {m} | {r['n_pairs']} | {pts(r['estimate_pts'])} | "
                 f"[{pts(r['ci95_low'])}, {pts(r['ci95_high'])}] | {r['ci95_high']:.1f} |")
    L += ["", "## Notes", ""]
    L += [f"- Re-detected `{n}`: undetected {x} → {y}, {c} rows changed"
          for n, x, y, c in outputs.redetected]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "REVIEWER_CHECKS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    pd.DataFrame(wrong_rows).to_csv(out_dir / "wrong_answers.csv", index=False)
    pd.DataFrame(tests).to_csv(out_dir / "template_tests.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
