"""Sweep the scorer switches against MBBQ Table 3 in all four languages.

Usage (from the fork root):
  python my_research/scorer/reproduce_table3.py --model mistral \
      --results my_research/results --data data
Expects pickles named <model>_<lang>[_control].pkl for en, nl, es, tr.

English is the primary gate: switch combinations are ranked by their English
error. nl, es and tr are a required check on the best combination.
"""
import argparse
import itertools
from pathlib import Path

from mbbq_scorer import annotate, attach_polarity, load_outputs, score, template_filter

# Table 3 [arXiv 2406.07243v3]. Accuracies in %, bias as fractions.
TABLE3 = {
    "mistral": {
        "n_templates": 56,
        "en": dict(acc_A=79.7, bias_A=0.0373, acc_D=75.5, bias_D=0.0054),
        "nl": dict(acc_A=73.4, bias_A=0.0503, acc_D=67.2, bias_D=0.0109),
        "es": dict(acc_A=76.3, bias_A=0.0691, acc_D=71.6, bias_D=0.0106),
        "tr": dict(acc_A=63.0, bias_A=0.0468, acc_D=44.0, bias_D=0.0454),
    },
    "zephyr": {
        "n_templates": 58,
        "en": dict(acc_A=50.1, bias_A=0.0853, acc_D=82.3, bias_D=0.0023),
        "nl": dict(acc_A=24.9, bias_A=0.1233, acc_D=76.3, bias_D=0.0366),
        "es": dict(acc_A=41.5, bias_A=0.1302, acc_D=68.5, bias_D=0.0384),
        "tr": dict(acc_A=39.9, bias_A=0.0400, acc_D=42.4, bias_D=0.0264),
    },
}
LANGS = ["en", "nl", "es", "tr"]
ACC_TOL, BIAS_TOL = 2.0, 0.01  # percentage points, fraction


def load(results, data, model, lang, control):
    name = f"{model}_{lang}{'_control' if control else ''}.pkl"
    df = load_outputs(Path(results) / name)
    return annotate(attach_polarity(df, data, lang, control=control))


def compare(s, t):
    """Error in tolerance units, and whether every metric is within tolerance."""
    diffs = {
        "acc_A": abs(100 * s["acc_A"] - t["acc_A"]) / ACC_TOL,
        "bias_A": abs(s["bias_A"] - t["bias_A"]) / BIAS_TOL,
        "acc_D": abs(100 * s["acc_D"] - t["acc_D"]) / ACC_TOL,
        "bias_D": abs(s["bias_D"] - t["bias_D"]) / BIAS_TOL,
    }
    return sum(diffs.values()), all(d <= 1 for d in diffs.values())


def fmt(s):
    return (f"acc_A={100*s['acc_A']:5.1f} bias_A={s['bias_A']:7.4f} "
            f"acc_D={100*s['acc_D']:5.1f} bias_D={s['bias_D']:7.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=TABLE3)
    ap.add_argument("--results", default="my_research/results")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    target = TABLE3[args.model]

    main_sets = {l: load(args.results, args.data, args.model, l, False) for l in LANGS}
    control_sets = {l: load(args.results, args.data, args.model, l, True) for l in LANGS}
    filters = {"main": template_filter(main_sets), "control": template_filter(control_sets)}

    combos = []
    for filter_on, bias_d, average in itertools.product(
            ["control", "main"], ["count", "acc_diff"], ["prompt_mean", "pooled"]):
        keep = filters[filter_on]
        per_lang = {}
        for lang in LANGS:
            s = score(main_sets[lang], bias_d=bias_d, average=average, templates=keep)
            err, ok = compare(s, target[lang])
            per_lang[lang] = (s, err, ok)
        combos.append(dict(filter_on=filter_on, bias_d=bias_d, average=average,
                           n=len(keep), langs=per_lang,
                           en_err=per_lang["en"][1],
                           all_err=sum(v[1] for v in per_lang.values())))

    print(f"Model: {args.model}   target #T = {target['n_templates']}")
    print("\nAll switch combinations, ranked by English error (primary gate):")
    for c in sorted(combos, key=lambda c: c["en_err"]):
        print(f"  en_err={c['en_err']:6.2f} all_err={c['all_err']:7.2f} "
              f"filter_on={c['filter_on']:7s} bias_d={c['bias_d']:8s} "
              f"average={c['average']:11s} #T={c['n']}")

    best = min(combos, key=lambda c: c["en_err"])
    print(f"\nBest by English: filter_on={best['filter_on']} bias_d={best['bias_d']} "
          f"average={best['average']} #T={best['n']}")
    for lang in LANGS:
        s, err, ok = best["langs"][lang]
        role = "PRIMARY " if lang == "en" else "required"
        print(f"  {lang} [{role}] {'PASS' if ok else 'FAIL'}  err={err:5.2f}  ours: {fmt(s)}")
        t = target[lang]
        print(f"  {' ' * 26}paper: acc_A={t['acc_A']:5.1f} bias_A={t['bias_A']:7.4f} "
              f"acc_D={t['acc_D']:5.1f} bias_D={t['bias_D']:7.4f}")

    best_all = min(combos, key=lambda c: c["all_err"])
    same = all(best[k] == best_all[k] for k in ("filter_on", "bias_d", "average"))
    print(f"\nSame combination wins on all-language error: {'yes' if same else 'NO'}")
    gate = (best["n"] == target["n_templates"]
            and all(best["langs"][l][2] for l in LANGS))
    print(f"GATE: {'PASS' if gate else 'FAIL'} (#T match and all 4 languages within tolerance)")


if __name__ == "__main__":
    main()
