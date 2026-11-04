"""Does feminine agreement move answers more when it agrees with the
stereotype? Exploratory for Mistral and Zephyr (DECISIONS.md, "Exploratory,
after the results"), planned for Qwen and T-lite ("Replication on two more
models").

  python3 my_research/scorer/fem_congruence.py [--results my_research/results]
      [--out my_research/results/fem]

The feminine cue points to the woman. A question is congruent if its
stereotyped answer (the `corrected` sign: the target on a negative question,
the other person on a non-negative one) is the woman, incongruent otherwise;
each template has one of each. Compares `fem` with `masc` (and, as the noise
floor, `masc_rerun` with `masc`) on the `fem` items, leaving out questions
that are the same in both sets. Writes CONGRUENCE.md to --out.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fem_analysis import (DATA_FILE, KEY, MODELS, classify, fmt_p, option_genders,  # noqa: E402
                          sign_test)
from mbbq_scorer import NO_TARGET, flip_target_loc  # noqa: E402
from score_both_signs import Outputs  # noqa: E402

SETS = ["masc", "fem", "masc_rerun"]
PAIRS = [("fem", "masc", "fem vs. masc"), ("masc_rerun", "masc", "noise floor: masc_rerun vs. masc")]
GENDER_NAME = {"f": "woman", "m": "man"}


def questions(path):
    return {json.loads(l)["example_id"]: json.loads(l)["question"]
            for l in open(path, encoding="utf-8")}


def with_stereotype(df, genders):
    """Add stereo_gender (gender of the stereotyped answer) and target_gender."""
    assert (df["target_loc"] != NO_TARGET).all()
    stereo = flip_target_loc(df["target_loc"], df["unknown_loc"], df["question_polarity"])
    g = df["example_id"].map(genders)
    return df.assign(stereo_gender=[gs[int(p)] for p, gs in zip(stereo, g)],
                     target_gender=[gs[int(t)] for t, gs in zip(df["target_loc"], g)])


def load(results):
    sets = {}
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        gen = subprocess.run([sys.executable, str(ROOT / "my_research/templates/generate_ru.py"),
                              "--out", str(data)], capture_output=True, text=True)
        if gen.returncode:
            sys.exit(f"generate_ru.py failed:\n{gen.stdout}{gen.stderr}")
        outputs = Outputs(results, ROOT / "data", data)
        q_fem, q_masc = questions(data / DATA_FILE["fem"]), questions(data / DATA_FILE["masc"])
        same_q = {e for e, q in q_fem.items() if q == q_masc[e]}
        for model in MODELS:
            for s in SETS:
                raw = outputs.get(model, "ru", s)
                if raw is None:
                    continue
                genders = option_genders(data / DATA_FILE[s])
                df = with_stereotype(classify(raw, genders), genders)
                sets[(model, s)] = df[df["example_id"].isin(set(q_fem) - same_q)]
    return sets, len(same_q), outputs.missing


def ambiguous(a, b):
    m = a.merge(b, on=KEY, suffixes=("_a", "_b"))
    m = m[(m["context_condition_a"] == "ambig") & m["pick_a"].notna() & m["pick_b"].notna()]
    m = m.assign(cong=m["stereo_gender_a"] == "f",
                 d=(m["pick_a"] == "f").astype(int) - (m["pick_b"] == "f").astype(int))
    rows = []
    for cong, x in m.groupby("cong"):
        pos, neg, p = sign_test(x.groupby(["subset", "example_id"])["d"].sum())
        rows.append(dict(kind="congruent" if cong else "incongruent", rows=len(x),
                         woman_b=(x["pick_b"] == "f").mean(), woman_a=(x["pick_a"] == "f").mean(),
                         stereo_b=(x["pick_b"] == x["stereo_gender_a"]).mean(),
                         stereo_a=(x["pick_a"] == x["stereo_gender_a"]).mean(),
                         up=pos, down=neg, p=p))
    per = m.groupby(["q_id_a", "cong"])["d"].mean().unstack()
    per = per.dropna()
    diff = per[True] - per[False]
    nz = diff[diff != 0]
    test = dict(n=len(diff), cong_bigger=int((diff > 0).sum()), incong_bigger=int((diff < 0).sum()),
                mean_cong=per[True].mean(), mean_incong=per[False].mean(),
                min_diff=diff.min(), max_diff=diff.max(),
                p=wilcoxon(nz).pvalue if len(nz) else 1.0)
    by_target = []
    for (t, cong), x in m.groupby(["target_gender_a", "cong"]):
        by_target.append(dict(target=GENDER_NAME.get(t, t), kind="congruent" if cong else "incongruent",
                              templates=x["q_id_a"].nunique(), rows=len(x),
                              woman_b=(x["pick_b"] == "f").mean(), woman_a=(x["pick_a"] == "f").mean()))
    overall = dict(stereo_b=(m["pick_b"] == m["stereo_gender_a"]).mean(),
                   stereo_a=(m["pick_a"] == m["stereo_gender_a"]).mean(),
                   anti_b=((m["pick_b"] != m["stereo_gender_a"]) & (m["pick_b"] != "u")).mean(),
                   anti_a=((m["pick_a"] != m["stereo_gender_a"]) & (m["pick_a"] != "u")).mean())
    return rows, test, by_target, overall


def disambiguated(a, b):
    m = a.merge(b, on=KEY, suffixes=("_a", "_b"))
    m = m[(m["context_condition_a"] == "disambig") & m["pick_a"].notna() & m["pick_b"].notna()]
    out = []
    for (gold, consistent), x in m.groupby([m["gold_gender_a"], m["gold_gender_a"] == m["stereo_gender_a"]]):
        ok_a, ok_b = x["pick_a"] == x["gold_gender_a"], x["pick_b"] == x["gold_gender_a"]
        d = (ok_a.astype(int) - ok_b.astype(int)).groupby([x["subset"], x["example_id"]]).sum()
        pos, neg, p = sign_test(d)
        out.append(dict(gold=GENDER_NAME[gold], context="consistent" if consistent else "inconsistent",
                        rows=len(x), acc_b=ok_b.mean(), acc_a=ok_a.mean(), up=pos, down=neg, p=p))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=ROOT / "my_research" / "results")
    ap.add_argument("--out", default=ROOT / "my_research" / "results" / "fem")
    args = ap.parse_args()
    sets, n_same, missing = load(args.results)
    L = ["# Exploratory: feminine agreement vs. the stereotype", "",
         "Generated by `my_research/scorer/fem_congruence.py`. Post hoc (defined in `DECISIONS.md`",
         "after the primary results), so exploratory. A question is *congruent* if its",
         "stereotyped answer (`corrected` sign) is the woman, so the feminine cue and the",
         f"stereotype point the same way. {n_same} items whose question is the same under `fem`",
         "and `masc` (q2 non-negative) are left out. «Woman» shares are of detected answers;",
         "items up / down = items with more / fewer «woman» answers (ambiguous) or correct",
         "answers (disambiguated) under the first set; sign tests over items.", ""]
    for model in MODELS:
        for a, b, title in PAIRS:
            if (model, a) not in sets or (model, b) not in sets:
                continue
            A, B = sets[(model, a)], sets[(model, b)]
            rows, test, by_target, overall = ambiguous(A, B)
            L += [f"## {model.capitalize()}, {title}", "", "### Ambiguous contexts", "",
                  f"| Question | Rows | «Woman» {b} → {a} | Stereotyped answer {b} → {a} | Items up / down | p |",
                  "|---|---|---|---|---|---|"]
            for r in rows:
                L.append(f"| {r['kind']} | {r['rows']} | {r['woman_b']:.3f} → {r['woman_a']:.3f} | "
                         f"{r['stereo_b']:.3f} → {r['stereo_a']:.3f} | {r['up']} / {r['down']} | "
                         f"{fmt_p(r['p'])} |")
            L += ["", f"- Shift toward «woman», congruent minus incongruent, per template: larger "
                      f"when congruent in {test['cong_bigger']} of {test['n']} templates, smaller in "
                      f"{test['incong_bigger']}; mean shift {test['mean_cong']:+.3f} (congruent) vs. "
                      f"{test['mean_incong']:+.3f} (incongruent); per-template difference "
                      f"{test['min_diff']:+.3f} to {test['max_diff']:+.3f}; Wilcoxon p = {fmt_p(test['p'])}.",
                  f"- All ambiguous answers: stereotyped {overall['stereo_b']:.3f} → "
                  f"{overall['stereo_a']:.3f}, anti-stereotyped {overall['anti_b']:.3f} → "
                  f"{overall['anti_a']:.3f}.", "",
                  "By the template's target gender:", "",
                  f"| Target | Question | Templates | Rows | «Woman» {b} → {a} |", "|---|---|---|---|---|"]
            for r in by_target:
                L.append(f"| {r['target']} | {r['kind']} | {r['templates']} | {r['rows']} | "
                         f"{r['woman_b']:.3f} → {r['woman_a']:.3f} |")
            L += ["", "### Disambiguated contexts", "",
                  f"| Gold | Context | Rows | Accuracy {b} → {a} | Items up / down | p |",
                  "|---|---|---|---|---|---|"]
            for r in disambiguated(A, B):
                L.append(f"| {r['gold']} | stereotype-{r['context']} | {r['rows']} | "
                         f"{r['acc_b']:.3f} → {r['acc_a']:.3f} | {r['up']} / {r['down']} | "
                         f"{fmt_p(r['p'])} |")
            L.append("")
    if missing:
        L.append("Missing, skipped: " + ", ".join(f"`{m}`" for m in missing))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "CONGRUENCE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
