"""Score every existing output under both bias signs, `published` and `corrected`.

  python3 my_research/scorer/score_both_signs.py [--results my_research/results]
      [--data data] [--out my_research/results/sign]

Configs, for Mistral and Zephyr:
- A, Table 3: en, nl, es, tr main sets on the frozen per-model template
  filter (control accuracy above chance in all four languages);
- B, five languages: en, nl, es, tr, ru main sets and the ru control set, all
  98 templates, no filter;
- C, gender ablation: ru against ru_masc on the 18 Gender_identity templates
  with a masc row, on the items (and prompts) present in both.

Everything else is the frozen method (README.md). Russian data is regenerated
from the worksheets into a temporary folder for the polarity join, and Russian
answers are re-detected with this checkout's detection code (Mistral's stored
detection predates be026e8); other languages keep their stored detection, as
in the Table 3 reproduction. No model is run.

Writes to --out: scores_<sign>.csv, per_template_<sign>.csv and
SIGN_COMPARISON.md. Stops if accuracy, the undetected counts or the template
filter differ between the signs.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "my_research" / "generation")]
import check_pickle  # noqa: E402  (Russian data and re-detection, as the pickle health check)
from mbbq_scorer import (SIGNS, annotate, attach_polarity, load_outputs, score,  # noqa: E402
                         score_per_template, template_filter)

MODELS = ["mistral", "zephyr"]
MBBQ_LANGS = ["en", "nl", "es", "tr"]
METRICS = ["acc_A", "bias_A", "acc_D", "bias_D"]
INVARIANT = ["acc_A", "acc_D", "n_undetected"]
CONTEXT = {"A": "ambig", "D": "disambig"}
# Gender_identity leak sets (lang "ru"): set -> run language. "masc_rerun" and
# "masc_batch" are second runs of masc (fem_analysis.py's noise floors), saved
# as <model>_ru_masc_rerun.pkl and <model>_ru_masc_batch.pkl.
LEAK_SETS = {"masc": "ru_masc", "fem": "ru_fem", "masc_rerun": "ru_masc", "masc_batch": "ru_masc"}


class Outputs:
    """Polarity-joined outputs by (model, lang, set); None if the pickle is missing."""

    def __init__(self, results, data, ru_data):
        self.results, self.data, self.ru_data = Path(results), Path(data), Path(ru_data)
        self.cache, self.missing, self.redetected = {}, [], []
        self.detect = check_pickle.detector()[0]

    def get(self, model, lang, set_):
        key = (model, lang, set_)
        if key not in self.cache:
            self.cache[key] = self._load(model, lang, set_)
        return self.cache[key]

    def _load(self, model, lang, set_):
        run_lang = LEAK_SETS.get(set_, lang)
        rerun = set_[len("masc"):] if set_.startswith("masc_") else ""  # "_rerun", "_batch"
        name = f"{model}_{run_lang}{'_control' if set_ == 'control' else ''}{rerun}.pkl"
        path = self.results / name
        if not path.exists():
            self.missing.append(name)
            return None
        df = load_outputs(path)
        if lang == "ru":
            new = pd.to_numeric(df.apply(lambda r: self.detect(r["model_ans"], r["answer"],
                                                               r["unknown_loc"], run_lang),
                                         axis=1), errors="coerce")
            old = df["answer_detected"]
            changed = ~((old == new) | (old.isna() & new.isna()))
            self.redetected.append((name, int(old.isna().sum()), int(new.isna().sum()),
                                    int(changed.sum())))
            df = df.assign(answer_detected=new)
        data = self.ru_data if lang == "ru" else self.data
        return attach_polarity(df, data, run_lang, control=set_ == "control")


def counts(df, templates=None):
    """{"A"|"D": (rows, undetected)} over the scored rows."""
    if templates is not None:
        df = df[[k in templates for k in zip(df["subset"], df["q_id"])]]
    out = {}
    for k, cond in CONTEXT.items():
        d = df[df["context_condition"] == cond]
        out[k] = (len(d), int((~d["detected"]).sum()))
    return out


def detected_per_template(df):
    d = df[df["detected"]]
    return {k: d[d["context_condition"] == cond].groupby(["subset", "q_id"]).size()
            for k, cond in CONTEXT.items()}


def same(a, b):
    return a.reset_index(drop=True).equals(b.reset_index(drop=True))


def score_set(raw, templates=None):
    """Both signs for one set: {sign: (means, per_template, counts, detected per template)}."""
    out = {}
    for sign in SIGNS:
        df = annotate(raw, sign=sign)
        out[sign] = (score(df, templates), score_per_template(df, templates),
                     counts(df, templates), detected_per_template(df))
    pub, cor = out["published"], out["corrected"]
    for m in INVARIANT:
        if not same(pub[1][m], cor[1][m]):
            sys.exit(f"STOP: {m} differs between the signs")
    if pub[2] != cor[2]:
        sys.exit("STOP: row or undetected counts differ between the signs")
    return out


def filter_both_signs(outputs, model):
    keeps = {}
    for sign in SIGNS:
        ctrl = {l: outputs.get(model, l, "control") for l in MBBQ_LANGS}
        if any(v is None for v in ctrl.values()):
            return None
        keeps[sign] = template_filter({l: annotate(v, sign=sign) for l, v in ctrl.items()})
    if keeps["published"] != keeps["corrected"]:
        sys.exit(f"STOP: {model} template filter differs between the signs")
    return keeps["published"]


def run(outputs):
    """[(model, config, lang, set, both-sign result)]."""
    jobs = []
    for model in MODELS:
        keep = filter_both_signs(outputs, model)
        if keep is None:
            print(f"{model}: control sets missing, config A skipped")
        else:
            print(f"{model}: config A filter keeps {len(keep)} templates")
            for lang in MBBQ_LANGS:
                raw = outputs.get(model, lang, "main")
                if raw is not None:
                    jobs.append((model, "A", lang, "main", score_set(raw, keep)))
        for lang, set_ in [(l, "main") for l in MBBQ_LANGS + ["ru"]] + [("ru", "control")]:
            raw = outputs.get(model, lang, set_)
            if raw is not None:
                jobs.append((model, "B", lang, set_, score_set(raw)))
        ru, masc = outputs.get(model, "ru", "main"), outputs.get(model, "ru", "masc")
        if ru is not None and masc is not None:
            items = set(zip(masc["subset"], masc["example_id"], masc["prompt_id"]))
            paired = ru[[k in items for k in zip(ru["subset"], ru["example_id"], ru["prompt_id"])]]
            back = set(zip(paired["subset"], paired["example_id"], paired["prompt_id"]))
            masc = masc[[k in back for k in zip(masc["subset"], masc["example_id"], masc["prompt_id"])]]
            print(f"{model}: config C pairs {len(paired)} ru rows with {len(masc)} ru_masc rows")
            jobs.append((model, "C", "ru", "main", score_set(paired)))
            jobs.append((model, "C", "ru", "masc", score_set(masc)))
    return jobs


def tables(jobs):
    scores, per_t = {s: [] for s in SIGNS}, {s: [] for s in SIGNS}
    for model, config, lang, set_, res in jobs:
        for sign in SIGNS:
            means, per, cnt, det = res[sign]
            for m in METRICS:
                rows, und = cnt[m[-1]]
                scores[sign].append(dict(model=model, config=config, lang=lang, set=set_, sign=sign,
                                         metric=m, value=means[m], n_templates=len(per),
                                         n_rows=rows, n_undetected=und))
            for (subset, q), r in per.iterrows():
                for m in METRICS:
                    n = int(det[m[-1]].get((subset, q), 0))
                    per_t[sign].append(dict(model=model, config=config, lang=lang, set=set_,
                                            subset=subset, q_id=q, metric=m, value=r[m], n_rows=n))
    return ({s: pd.DataFrame(v) for s, v in scores.items()},
            {s: pd.DataFrame(v) for s, v in per_t.items()})


def comparison_md(scores, outputs):
    both = pd.concat(scores.values())
    wide = both.pivot_table(index=["model", "config", "lang", "set"], columns=["metric", "sign"],
                            values="value", sort=False)
    nt = both.groupby(["model", "config", "lang", "set"], sort=False)["n_templates"].first()
    title = {"A": "A. Table 3 (frozen template filter)",
             "B": "B. Five languages, all templates, no filter",
             "C": "C. Gender ablation: ru vs ru_masc, paired items"}
    out = ["# Bias scores under both signs", "",
           "Generated by `my_research/scorer/score_both_signs.py`; numbers from",
           "`scores_published.csv` and `scores_corrected.csv`. Accuracies in %, biases as",
           "fractions. `published`: the target group's answer is biased on every question",
           "(released code, Table 3). `corrected`: the target on a negative question, the",
           "other person on a non-negative one (BBQ's definition). Accuracies don't depend",
           "on the sign. Russian answers re-detected with the current detection code.", ""]
    for model in MODELS:
        for config in "ABC":
            keys = [k for k in wide.index if k[0] == model and k[1] == config]
            if not keys:
                continue
            out += [f"## {model.capitalize()}, {title[config]}", "",
                    "| lang | #T | Acc_A | Acc_D | Bias_A pub | Bias_A corr | Bias_D pub | Bias_D corr |",
                    "|---|---|---|---|---|---|---|---|"]
            for k in keys:
                r = wide.loc[k]
                label = k[2] + ("" if k[3] == "main" else f" {k[3]}")
                out.append(f"| {label} | {nt[k]} | {100 * r[('acc_A', 'published')]:.1f} | "
                           f"{100 * r[('acc_D', 'published')]:.1f} | "
                           + " | ".join(f"{r[(m, s)]:.4f}" for m in ("bias_A", "bias_D")
                                        for s in SIGNS) + " |")
            out.append("")
            ranked = [k for k in keys if config == "C" or k[3] == "main"]
            for sign in SIGNS:
                order = sorted(ranked, key=lambda k: -wide.loc[k][("bias_A", sign)])
                names = [k[2] + ("" if k[3] == "main" else f" {k[3]}") for k in order]
                out.append(f"- Bias_A ranking, {sign} (most biased first): " + " > ".join(names))
            out.append("")
    notes = ["## Notes", ""]
    if outputs.redetected:
        notes.append("Russian re-detection (undetected stored → current, rows changed):")
        notes += [f"- `{n}`: {a} → {b}, {c} rows changed" for n, a, b, c in outputs.redetected]
    if outputs.missing:
        notes.append("Missing outputs, skipped: " + ", ".join(f"`{m}`" for m in outputs.missing))
    return "\n".join(out + notes) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=ROOT / "my_research" / "results")
    ap.add_argument("--data", default=ROOT / "data")
    ap.add_argument("--out", default=ROOT / "my_research" / "results" / "sign")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        ru_data = Path(tmp) / "data"
        ru_data.mkdir()
        gen = subprocess.run(
            [sys.executable, str(ROOT / "my_research/templates/generate_ru.py"), "--out", str(ru_data)],
            capture_output=True, text=True)
        if gen.returncode:
            sys.exit(f"generate_ru.py failed:\n{gen.stdout}{gen.stderr}")
        outputs = Outputs(args.results, args.data, ru_data)
        jobs = run(outputs)
    scores, per_t = tables(jobs)
    for sign in SIGNS:
        scores[sign].to_csv(out / f"scores_{sign}.csv", index=False)
        per_t[sign].to_csv(out / f"per_template_{sign}.csv", index=False)
    (out / "SIGN_COMPARISON.md").write_text(comparison_md(scores, outputs), encoding="utf-8")
    for n, a, b, c in outputs.redetected:
        print(f"re-detected {n}: undetected {a} -> {b}, {c} rows changed")
    if outputs.missing:
        print("missing, skipped:", ", ".join(outputs.missing))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
