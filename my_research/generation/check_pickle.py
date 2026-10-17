"""Health check for a pickle written by run_step6.py (any language).

  python my_research/generation/check_pickle.py my_research/results/mistral_ru.pkl \
      --model mistral --lang ru [--control] [--data DIR] [--redetected OUT.pkl]

Checks, in order:
1. rows: per subset and context, 5 prompts per item, no duplicate rows;
2. data: the columns mbbq.py derives from the data (and the options) equal a
   fresh get_samples. For ru/ru_masc the data is regenerated from the current
   worksheets into a temporary folder unless --data is given; other languages
   read data/;
3. answers: missing, empty, repeated verbatim, prompt-format leftovers, share
   mostly in Latin script;
4. lengths: answer tokens against the model's max_new_tokens (truncated
   answers), with the pinned tokenizer from the local Hugging Face cache;
5. detection: stored undetected rate (by subset, prompt, context), and the
   stored detection against the detection code in this checkout: every
   detected row that changes is printed. --redetected saves the re-detected
   column. answer_processed is checked against score_answer;
6. scores: frozen scorer on all templates of this pickle, next to the same
   model's en/nl/es/tr pickles in the same folder (also all templates, no
   control filter, so not Table 3's numbers).

To compare the stored detection with a commit's code, run this file from an
export of that commit (git archive) with an absolute pickle path.
"""
import argparse
import ast
import contextlib
import io
import os
import string
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / "my_research" / "detection"),
                str(ROOT / "my_research" / "generation"), str(ROOT / "my_research" / "scorer")]

import answer_detection as ad  # noqa: E402  (upstream)
import ru_detection  # noqa: E402
from ask_model_paper import PAPER_MODELS  # noqa: E402
from mbbq_scorer import annotate, attach_polarity, score  # noqa: E402
from run_step6 import KEY_COLS, SUBSETS  # noqa: E402

ru_detection.register(ad)
N_PROMPTS = 5


def mbbq_functions(*names):
    """Functions from upstream mbbq.py, without running its imports of the model stack."""
    tree = ast.parse((ROOT / "mbbq.py").read_text())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"string": string, "pd": pd, "np": __import__("numpy"), "json": __import__("json"),
          "tqdm": __import__("tqdm").tqdm,
          **{k: v for k, v in vars(ad).items() if not k.startswith("__")}}
    exec(compile(ast.Module(fns, []), "mbbq.py", "exec"), ns)
    return ns


def fresh_samples(lang, control, subsets, data, tmp):
    """get_samples on `data`; Russian data is regenerated into `tmp` if `data` is None."""
    ns = mbbq_functions("get_samples")
    if data is None and lang in ru_detection.LANGS:
        data = Path(tmp) / "data"
        data.mkdir()
        out = subprocess.run([sys.executable, str(ROOT / "my_research/templates/generate_ru.py"),
                              "--out", str(data)], capture_output=True, text=True)
        if out.returncode:
            sys.exit(f"generate_ru.py failed:\n{out.stdout}{out.stderr}")
    data = Path(data or ROOT / "data").resolve()
    work = Path(tmp) / "work"
    work.mkdir()
    (work / "data").symlink_to(data)
    cwd = os.getcwd()
    os.chdir(work)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return ns["get_samples"](subsets, control=control, language=lang), data
    finally:
        os.chdir(cwd)


def detector():
    ns = mbbq_functions("auto_detect", "check_all", "detect_answer", "score_answer")
    return ru_detection.wrap_detect_answer(ns["detect_answer"]), ns["score_answer"]


def latin_share(text):
    lat = sum(c.isascii() and c.isalpha() for c in text)
    cyr = sum("Ѐ" <= c <= "ӿ" for c in text)
    return lat / (lat + cyr) if lat + cyr else float("nan")


def short(text, n=160):
    return text.strip().replace("\n", " ")[:n]


def head(title):
    print(f"\n== {title}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pickle")
    ap.add_argument("--model", required=True, choices=PAPER_MODELS)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--data", help="folder with the data the run used (default: see above)")
    ap.add_argument("--redetected", help="save the re-detected answer column here")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        check(args, tmp)


def check(args, tmp):
    cfg = PAPER_MODELS[args.model]
    df = pd.read_pickle(args.pickle)
    subsets = [s for s in SUBSETS if s in set(df["subset"])]

    head("1. Rows")
    print(f"{len(df)} rows, {len(df) // N_PROMPTS} items x {N_PROMPTS} prompts; columns: {list(df.columns)}")
    print(df.groupby(["subset", "context_condition"]).size().unstack().to_string())
    per_item = df.groupby(["subset", "example_id"])["prompt_id"].agg(["size", "nunique"])
    print(f"items without exactly {N_PROMPTS} distinct prompts: "
          f"{((per_item['size'] != N_PROMPTS) | (per_item['nunique'] != N_PROMPTS)).sum()}")
    print(f"duplicate (subset, example_id, prompt_id): "
          f"{df.duplicated(['subset', 'example_id', 'prompt_id']).sum()}")
    print(f"target_loc = 100 (no single target answer; scorer drops): {(df['target_loc'] == 100).sum()}")

    head("2. Data")
    ref, data = fresh_samples(args.lang, args.control, subsets, args.data, tmp)
    print(f"fresh get_samples from {data}: {len(ref)} rows")
    try:
        pd.testing.assert_frame_equal(df[KEY_COLS].reset_index(drop=True),
                                      ref[KEY_COLS].reset_index(drop=True))
        print(f"identical in {KEY_COLS}")
    except AssertionError as e:
        print(f"DIFFERENT: {str(e)[:600]}")
    same_opts = (df["model_ans"].reset_index(drop=True) == ref["model_ans"].reset_index(drop=True)).all() \
        if len(df) == len(ref) else False
    print(f"options (model_ans) identical: {same_opts}")

    head("3. Answers")
    ans = df["answer"]
    is_str = ans.map(lambda a: isinstance(a, str))
    print(f"not a string: {(~is_str).sum()}; empty or whitespace: {(ans[is_str].str.strip() == '').sum()}")
    counts = ans[is_str].str.strip().value_counts()
    rep = counts[counts > 1]
    print(f"distinct answers: {len(counts)}; texts repeated verbatim: {len(rep)} "
          f"(covering {rep.sum()} rows); most repeated:")
    for text, n in rep.head(5).items():
        print(f"   {n:5d}x  {short(text, 120)!r}")
    for mark in ["[INST]", "[/INST]", "<s>", "</s>", "<|user|>", "<|assistant|>", "�"]:
        n = ans[is_str].str.contains(mark, regex=False).sum()
        if n:
            print(f"answers containing {mark!r}: {n}")
    lat = ans[is_str].map(latin_share)
    print(f"answers mostly in Latin script (>50% of letters): {(lat > 0.5).mean():.1%}; "
          f"by prompt: " + ", ".join(f"{p}: {(lat[df['prompt_id'] == p] > 0.5).mean():.1%}"
                                     for p in sorted(df["prompt_id"].unique())))

    head("4. Lengths")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["hf_model"], revision=cfg["revision"],
                                        local_files_only=True)
    n_tok = pd.Series([len(t) for t in tok(ans.where(is_str, "").tolist(),
                                           add_special_tokens=False)["input_ids"]], index=df.index)
    cap = cfg["max_new_tokens"]
    print(f"answer tokens (re-tokenized, may differ from the generated count by a few): "
          f"median {n_tok.median():.0f}, p99 {n_tok.quantile(0.99):.0f}, max {n_tok.max()}; "
          f"cap {cap}")
    print(f"answers with >= {cap - 5} tokens (likely truncated): {(n_tok >= cap - 5).sum()}")
    q_tok = pd.Series([len(t) for t in tok([cfg["prompt"].format(q) for q in df["question"]])["input_ids"]])
    print(f"prompt tokens: median {q_tok.median():.0f}, max {q_tok.max()}")

    head("5. Detection")
    det = df["answer_detected"]
    print(f"stored undetected: {det.isna().sum()} of {len(df)} ({det.isna().mean():.2%})")
    for col in ["subset", "prompt_id", "context_condition"]:
        g = det.isna().groupby(df[col])
        print(f"   by {col}: " + ", ".join(f"{k}: {v}" for k, v in g.sum().items()))
    detect, score_answer = detector()
    new = pd.to_numeric(df.apply(lambda r: detect(r["model_ans"], r["answer"], r["unknown_loc"],
                                                  args.lang), axis=1), errors="coerce")
    same = (det == new) | (det.isna() & new.isna())
    print(f"re-detected with this checkout ({ROOT}): undetected {new.isna().sum()} "
          f"({new.isna().mean():.2%}); rows differing from stored: {(~same).sum()} "
          f"(newly detected {(det.isna() & new.notna()).sum()}, "
          f"newly undetected {(det.notna() & new.isna()).sum()}, "
          f"detected answer changed {(det.notna() & new.notna() & ~same).sum()})")
    for i in df.index[det.notna() & ~same]:
        r = df.loc[i]
        print(f"   [{i}] {det[i]} -> {new[i]}  unk={r.unknown_loc} {r.model_ans}  {short(r.answer)!r}")
    if args.redetected:
        pd.to_pickle(new, args.redetected)
        print(f"saved to {args.redetected}")
    proc = df.apply(lambda r: score_answer(r["answer_detected"], r["unknown_loc"], r["target_loc"]), axis=1)
    ok = (proc == df["answer_processed"]) | (proc.isna() & df["answer_processed"].isna())
    print(f"answer_processed = score_answer(stored detection): {ok.all()} ({(~ok).sum()} differ)")

    head("6. Scores (all templates, no control filter)")
    metrics = ["acc_A", "bias_A", "acc_D", "bias_D", "n_undetected"]
    rows = {}
    suffix = "_control" if args.control else ""
    try:
        rows[f"{args.lang} (stored)"] = score(annotate(attach_polarity(df, data, args.lang, args.control)))
        rows[f"{args.lang} (re-detected)"] = score(annotate(attach_polarity(
            df.assign(answer_detected=new), data, args.lang, args.control)))
    except AssertionError as e:
        print(f"polarity join failed: {e}")
    results = Path(args.pickle).resolve().parent
    for lang in ["en", "nl", "es", "tr"]:
        p = results / f"{args.model}_{lang}{suffix}.pkl"
        if p.exists():
            other = pd.read_pickle(p)
            other = other[other["subset"].isin(subsets)]
            rows[lang] = score(annotate(attach_polarity(other, ROOT / "data", lang, args.control)))
    print(pd.DataFrame(rows).T[metrics].to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
