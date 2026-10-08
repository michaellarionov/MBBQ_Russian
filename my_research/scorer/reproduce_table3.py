"""Sweep the scorer switches against MBBQ Table 3 and report the closest match.

Usage (from the fork root):
  python my_research/scorer/reproduce_table3.py --model mistral \
      --results my_research/results --data data
Expects pickles named <model>_<lang>[_control].pkl for en, nl, es, tr.
"""
import argparse
import itertools
from pathlib import Path

from mbbq_scorer import annotate, attach_polarity, load_outputs, score, template_filter

# Table 3, English rows [arXiv 2406.07243v3]. Accuracies in %, bias as fractions.
TABLE3_EN = {
    "mistral": dict(acc_A=79.7, bias_A=0.0373, acc_D=75.5, bias_D=0.0054, n_templates=56),
    "zephyr": dict(acc_A=50.1, bias_A=0.0853, acc_D=82.3, bias_D=0.0023, n_templates=58),
}
LANGS = ["en", "nl", "es", "tr"]


def load(results, data, model, lang, control):
    name = f"{model}_{lang}{'_control' if control else ''}.pkl"
    df = load_outputs(Path(results) / name)
    return annotate(attach_polarity(df, data, lang, control=control))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=TABLE3_EN)
    ap.add_argument("--results", default="my_research/results")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    target = TABLE3_EN[args.model]

    main_sets = {l: load(args.results, args.data, args.model, l, False) for l in LANGS}
    control_sets = {l: load(args.results, args.data, args.model, l, True) for l in LANGS}
    filters = {"main": template_filter(main_sets), "control": template_filter(control_sets)}

    rows = []
    for filter_on, bias_d, average in itertools.product(
            ["control", "main"], ["count", "acc_diff"], ["prompt_mean", "pooled"]):
        keep = filters[filter_on]
        s = score(main_sets["en"], bias_d=bias_d, average=average, templates=keep)
        err = (abs(100 * s["acc_A"] - target["acc_A"]) / 2
               + abs(s["bias_A"] - target["bias_A"]) / 0.01
               + abs(100 * s["acc_D"] - target["acc_D"]) / 2
               + abs(s["bias_D"] - target["bias_D"]) / 0.01)
        rows.append((err, filter_on, bias_d, average, len(keep), s))

    print(f"Target ({args.model}, en): {target}")
    for err, filter_on, bias_d, average, n, s in sorted(rows, key=lambda r: r[0]):
        print(f"err={err:6.2f} filter_on={filter_on:7s} bias_d={bias_d:8s} "
              f"average={average:11s} #T={n:3d} acc_A={100*s['acc_A']:.1f} "
              f"bias_A={s['bias_A']:.4f} acc_D={100*s['acc_D']:.1f} bias_D={s['bias_D']:.4f}")


if __name__ == "__main__":
    main()
