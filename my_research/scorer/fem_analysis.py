"""Feminine vs. masculine question agreement on Gender_identity (the analysis
fixed in DECISIONS.md, "Feminine variant", before the run).

  python3 my_research/scorer/fem_analysis.py [--results my_research/results]
      [--out my_research/results/fem]

Reads, per model, <model>_ru.pkl, <model>_ru_masc.pkl, <model>_ru_fem.pkl and
<model>_ru_masc_rerun.pkl (missing ones are listed and skipped). Every set is
re-detected with the current detection code and restricted to the items of
the `fem` set (or of `masc` while `fem` is missing), all 5 prompts.

Each detected answer is classed woman / man / unknown from the English tags
(woman/girl/F, man/boy/M). Paired comparisons (a vs. b, same item and prompt):
- primary, ambiguous contexts: per item, (prompts answered «woman» under a)
  minus (under b), and the same for «man»; exact two-sided sign test over
  items with a nonzero difference (the item is the unit: its prompts aren't
  independent). Holm correction over the 4 primary tests (H1 woman, H2 man,
  fem vs. masc, × 2 models);
- secondary: Wilcoxon signed-rank over templates on the per-template mean
  difference; row-level discordant counts; disambiguated accuracy by the gold
  person's gender (sign tests over items);
- noise floor: masc_rerun vs. masc.
Writes FEM_RESULTS.md and fem_results.csv to --out.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mbbq_scorer import annotate, score  # noqa: E402
from score_both_signs import MODELS, Outputs  # noqa: E402

SETS = ["main", "masc", "fem", "masc_rerun"]
DATA_FILE = {"main": "Gender_identity_ru.jsonl", "masc": "Gender_identity_ru_masc.jsonl",
             "fem": "Gender_identity_ru_fem.jsonl", "masc_rerun": "Gender_identity_ru_masc.jsonl"}
GENDER = {"woman": "f", "girl": "f", "F": "f", "man": "m", "boy": "m", "M": "m", "unknown": "u"}
KEY = ["subset", "example_id", "prompt_id"]
# (a, b, role): a vs. b
PAIRS = [("fem", "masc", "primary"), ("masc_rerun", "masc", "noise floor"),
         ("fem", "masc_rerun", "secondary"), ("masc", "main", "descriptive"),
         ("fem", "main", "descriptive")]


def option_genders(path):
    """{example_id: (gender of ans0, ans1, ans2)} from a Russian data file."""
    out = {}
    for line in open(path, encoding="utf-8"):
        s = json.loads(line)
        # "o": the trans/cis templates' options, which this analysis doesn't compare
        out[s["example_id"]] = tuple(GENDER.get(s["answer_info"][f"ans{i}"][1], "o")
                                     for i in range(3))
    return out


def classify(df, genders):
    g = df["example_id"].map(genders)
    det = df["answer_detected"]
    df = df.copy()
    df["pick"] = [None if np.isnan(a) else gs[int(a)] for a, gs in zip(det, g)]
    df["gold_gender"] = [gs[int(l)] for l, gs in zip(df["labels"], g)]
    return df


def describe(df):
    a = df[df["context_condition"] == "ambig"]
    d = df[df["context_condition"] == "disambig"]
    a_det = a[a["pick"].notna()]
    out = {"ambig rows": len(a), "ambig undetected": int(a["pick"].isna().sum())}
    for g, name in (("f", "woman"), ("m", "man"), ("u", "unknown")):
        out[f"share {name}"] = (a_det["pick"] == g).mean()
    pub, cor = score(annotate(df)), score(annotate(df, sign="corrected"))
    out.update({"Acc_A": pub["acc_A"], "Bias_A pub": pub["bias_A"], "Bias_A corr": cor["bias_A"],
                "Acc_D": pub["acc_D"]})
    d_det = d[d["pick"].notna()]
    for g, name in (("f", "woman"), ("m", "man")):
        x = d_det[d_det["gold_gender"] == g]
        out[f"Acc_D gold {name}"] = (x["pick"] == x["gold_gender"]).mean()
    return out


def sign_test(diffs):
    pos, neg = int((diffs > 0).sum()), int((diffs < 0).sum())
    p = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else 1.0
    return pos, neg, p


def compare(a, b):
    """Paired a vs. b: {test name: values}."""
    m = a.merge(b, on=KEY, suffixes=("_a", "_b"))
    both = m[m["pick_a"].notna() & m["pick_b"].notna()]
    amb = both[both["context_condition_a"] == "ambig"]
    res = {"paired rows": len(m), "both detected": len(both),
           "answer changed": float((both["pick_a"] != both["pick_b"]).mean())}
    for g, name in (("f", "woman"), ("m", "man")):
        x = amb.assign(d=(amb["pick_a"] == g).astype(int) - (amb["pick_b"] == g).astype(int))
        item = x.groupby(["subset", "example_id"])["d"].sum()
        pos, neg, p = sign_test(item)
        tmpl = x.groupby(["subset", "q_id_a"])["d"].mean()
        nz = tmpl[tmpl != 0]
        w_p = wilcoxon(nz).pvalue if len(nz) else 1.0
        res[name] = {"rows a only": int((x["d"] > 0).sum()), "rows b only": int((x["d"] < 0).sum()),
                     "items a>b": pos, "items a<b": neg, "p sign test": p,
                     "templates a>b": int((tmpl > 0).sum()), "templates a<b": int((tmpl < 0).sum()),
                     "p Wilcoxon": w_p,
                     "share diff": float((amb["pick_a"] == g).mean() - (amb["pick_b"] == g).mean())}
    dis = both[both["context_condition_a"] == "disambig"]
    for g, name in (("f", "woman"), ("m", "man")):
        x = dis[dis["gold_gender_a"] == g]
        d = ((x["pick_a"] == x["gold_gender_a"]).astype(int)
             - (x["pick_b"] == x["gold_gender_a"]).astype(int))
        pos, neg, p = sign_test(d.groupby([x["subset"], x["example_id"]]).sum())
        res[f"Acc_D gold {name}"] = {"diff": float(d.mean()) if len(d) else float("nan"),
                                     "items a>b": pos, "items a<b": neg, "p sign test": p}
    return res


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def fmt_p(p):
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=ROOT / "my_research" / "results")
    ap.add_argument("--out", default=ROOT / "my_research" / "results" / "fem")
    args = ap.parse_args()
    out_dir = Path(args.out)
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        gen = subprocess.run([sys.executable, str(ROOT / "my_research/templates/generate_ru.py"),
                              "--out", str(data)],
                             capture_output=True, text=True)
        if gen.returncode:
            sys.exit(f"generate_ru.py failed:\n{gen.stdout}{gen.stderr}")
        outputs = Outputs(args.results, ROOT / "data", data)
        sets = {}
        for model in MODELS:
            for s in SETS:
                raw = outputs.get(model, "ru", s)
                if raw is not None:
                    raw = raw[raw["subset"] == "Gender_identity"]
                    sets[(model, s)] = classify(raw, option_genders(data / DATA_FILE[s]))

    lines, rows, primary = ["# Feminine vs. masculine question agreement (Gender_identity)", ""], [], []
    lines += ["Generated by `my_research/scorer/fem_analysis.py`; plan fixed in `DECISIONS.md`",
              "(\"Feminine variant\") before the run. Sets: `main` = `ru` (neutral question),",
              "`masc`, `fem`, `masc_rerun` (second run of `masc`, the noise floor). All sets",
              "re-detected, restricted to the same items × 5 prompts. Shares are of detected",
              "ambiguous answers.", ""]
    for model in MODELS:
        have = {s: sets[(model, s)] for s in SETS if (model, s) in sets}
        base = "fem" if "fem" in have else "masc"
        if base not in have:
            continue
        items = set(zip(have[base]["subset"], have[base]["example_id"]))
        have = {s: d[[k in items for k in zip(d["subset"], d["example_id"])]] for s, d in have.items()}
        n_items = len(items)
        lines += [f"## {model.capitalize()} ({n_items} items, {len(have[base])} rows per set)", "",
                  "| Set | Undetected | Woman | Man | Unknown | Acc_A | Bias_A pub | Bias_A corr "
                  "| Acc_D | Acc_D gold woman | Acc_D gold man |", "|" + "---|" * 11]
        for s, d in have.items():
            x = describe(d)
            rows += [dict(model=model, kind="set", a=s, b="", test=k, value=v) for k, v in x.items()]
            lines.append(f"| {s} | {x['ambig undetected']} | {x['share woman']:.3f} | "
                         f"{x['share man']:.3f} | {x['share unknown']:.3f} | {x['Acc_A']:.3f} | "
                         f"{x['Bias_A pub']:.4f} | {x['Bias_A corr']:.4f} | {x['Acc_D']:.3f} | "
                         f"{x['Acc_D gold woman']:.3f} | {x['Acc_D gold man']:.3f} |")
        lines += ["", "| a vs. b | Role | Answers changed | «Woman»: items a>b / a<b, p | "
                  "«Man»: items a>b / a<b, p | Templates «woman» a>b / a<b, Wilcoxon p | "
                  "Acc_D gold woman: diff, p | Acc_D gold man: diff, p |", "|" + "---|" * 8]
        for a, b, role in PAIRS:
            if a not in have or b not in have:
                continue
            r = compare(have[a], have[b])
            for k, v in r.items():
                if isinstance(v, dict):
                    rows += [dict(model=model, kind=role, a=a, b=b, test=f"{k}: {kk}", value=vv)
                             for kk, vv in v.items()]
                else:
                    rows.append(dict(model=model, kind=role, a=a, b=b, test=k, value=v))
            if role == "primary":
                primary += [(model, "woman", r["woman"]["p sign test"]),
                            (model, "man", r["man"]["p sign test"])]
            w, mn = r["woman"], r["man"]
            lines.append(
                f"| {a} vs. {b} | {role} | {r['answer changed']:.1%} | "
                f"{w['items a>b']} / {w['items a<b']}, {fmt_p(w['p sign test'])} | "
                f"{mn['items a>b']} / {mn['items a<b']}, {fmt_p(mn['p sign test'])} | "
                f"{w['templates a>b']} / {w['templates a<b']}, {fmt_p(w['p Wilcoxon'])} | "
                + " | ".join(f"{r[k]['diff']:+.3f}, {fmt_p(r[k]['p sign test'])}"
                             for k in ("Acc_D gold woman", "Acc_D gold man")) + " |")
        lines.append("")
    if primary:
        adj = holm([p for _, _, p in primary])
        lines += ["## Primary tests (fem vs. masc, ambiguous, sign test over items)", "",
                  "| Model | Hypothesis | p | Holm p |", "|---|---|---|---|"]
        for (model, h, p), q in zip(primary, adj):
            hyp = "H1: more «woman» under fem" if h == "woman" else "H2: more «man» under masc"
            lines.append(f"| {model} | {hyp} | {fmt_p(p)} | {fmt_p(q)} |")
            rows.append(dict(model=model, kind="primary", a="fem", b="masc",
                             test=f"{h}: Holm p", value=q))
        lines += ["", "H2 is tested as fem vs. masc «man» answers: a < b supports it.", ""]
    lines += ["## Notes", ""]
    lines += [f"- Re-detected `{n}`: undetected {x} → {y}, {c} rows changed"
              for n, x, y, c in outputs.redetected]
    if outputs.missing:
        lines.append("- Missing, skipped: " + ", ".join(f"`{m}`" for m in outputs.missing))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "FEM_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "fem_results.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
