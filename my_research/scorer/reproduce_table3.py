"""Score the frozen MBBQ method against Table 3 in all four languages.

Usage (from the fork root):
  python my_research/scorer/reproduce_table3.py --model mistral \
      --results my_research/results --data data
Expects pickles named <model>_<lang>[_control].pkl for en, nl, es, tr.

Prints our scores next to the paper's, the deviation of each metric, the
largest deviations, and the template count (#T) against the paper's.
"""
import argparse
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
METRICS = ("acc_A", "bias_A", "acc_D", "bias_D")


def load(results, data, model, lang, control):
    name = f"{model}_{lang}{'_control' if control else ''}.pkl"
    return annotate(attach_polarity(load_outputs(Path(results) / name), data, lang,
                                    control=control))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=TABLE3)
    ap.add_argument("--results", default="my_research/results")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    target = TABLE3[args.model]

    keep = template_filter({l: load(args.results, args.data, args.model, l, True)
                            for l in LANGS})
    print(f"Model: {args.model}   #T = {len(keep)} (paper: {target['n_templates']})")
    worst = {"acc": (0.0, ""), "bias": (0.0, "")}
    for lang in LANGS:
        s = score(load(args.results, args.data, args.model, lang, False), templates=keep)
        t = target[lang]
        ours = {m: 100 * s[m] if m.startswith("acc") else s[m] for m in METRICS}
        dev = {m: ours[m] - t[m] for m in METRICS}
        print(f"\n  {lang}  ours:  " + "  ".join(f"{m}={ours[m]:8.4f}" for m in METRICS))
        print("      paper: " + "  ".join(f"{m}={t[m]:8.4f}" for m in METRICS))
        print("      dev:   " + "  ".join(f"{m}={dev[m]:+8.4f}" for m in METRICS))
        for m in METRICS:
            kind = m.split("_")[0]
            if abs(dev[m]) > worst[kind][0]:
                worst[kind] = (abs(dev[m]), f"{lang} {m}")
    print(f"\nLargest accuracy deviation: {worst['acc'][0]:.2f} points ({worst['acc'][1]})")
    print(f"Largest bias deviation:     {worst['bias'][0]:.4f} ({worst['bias'][1]})")


if __name__ == "__main__":
    main()
