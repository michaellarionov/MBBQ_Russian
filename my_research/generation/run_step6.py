"""Step 6: generate MBBQ model outputs, resumably (built for Colab).

For each language x {main, control} x subset, runs upstream mbbq.py (through
run_mbbq.py) in --work_dir: generate_samples, ask_model, detect_answers.
ask_model runs one subset at a time, so a dropped session loses at most one
subset; finished steps are skipped on rerun (mbbq.py skips them itself).
get_samples is deterministic, so the per-subset pickles concatenate to the
same rows as one all-subsets run; this is checked against a fresh
generate_samples before anything is written to --results_dir.

Output: <results_dir>/<key>_<lang>[_control].pkl (what reproduce_table3.py
reads) and <key>_runinfo.json (model commit, versions, timings).

The model is pinned: the snapshot at --revision (default: current main,
recorded in runinfo) is downloaded and its local path is passed to mbbq.py.
Log in to Hugging Face first (huggingface-cli login or HF_TOKEN).

  python my_research/generation/run_step6.py --key mistral \
      --hf_model mistralai/Mistral-7B-Instruct-v0.2 \
      --work_dir /content/drive/MyDrive/mbbq/work \
      --results_dir /content/drive/MyDrive/mbbq/results
"""
import argparse
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_MBBQ = Path(__file__).resolve().parent / "run_mbbq.py"
SUBSETS = ["Age", "Disability_status", "Gender_identity",
           "Physical_appearance", "SES", "Sexual_orientation"]
LANGS = ["en", "nl", "es", "tr"]
# Columns mbbq.py derives from the data; must match a full-run reference.
KEY_COLS = ["subset", "example_id", "q_id", "question", "context_condition",
            "unknown_loc", "labels", "prompt_id", "target_loc"]
WEIGHTS = ["*.json", "tokenizer.model", "model-*.safetensors"]


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_atomic(obj, path):
    tmp = Path(f"{path}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def mbbq(work_dir, mode, lang, exp_id, control, subsets, model=None, bs=None):
    cmd = [sys.executable, str(RUN_MBBQ), "-mode", mode, "-lang", lang,
           "-exp_id", exp_id, "-subsets", *subsets]
    if control:
        cmd.append("--control")
    if model:
        cmd += ["-model", model]
    if bs:
        cmd += ["-bs", str(bs)]
    subprocess.run(cmd, cwd=work_dir, check=True)


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU: ask_model needs one (Colab: A100 or L4 runtime).")
    # is_bf16_supported() is True on a T4 too (emulated, very slow), so check
    # for Ampere or newer, which has native bf16.
    if torch.cuda.get_device_capability()[0] < 8:
        sys.exit(f"{torch.cuda.get_device_name()} has no native bf16 (T4?); "
                 "mbbq.py runs in bf16. Switch to an A100 or L4 runtime.")
    return torch.cuda.get_device_name()


def resolve_model(hf_model, revision, info):
    from huggingface_hub import model_info, snapshot_download
    sha = model_info(hf_model, revision=revision).sha
    if info["results"] and info.get("revision") not in (None, sha):
        sys.exit(f"Earlier runs used commit {info['revision']}, "
                 f"but {revision or 'main'} is {sha}. "
                 f"Pass --revision {info['revision']} to keep runs comparable.")
    return sha, snapshot_download(hf_model, revision=sha, allow_patterns=WEIGHTS)


def run_one(args, lang, control, model_path, info):
    tag = f"{args.key}_{lang}{'_control' if control else ''}"
    out = Path(args.results_dir) / f"{tag}.pkl"
    if out.exists():
        print(f"[{tag}] done already, skipping")
        return
    work = Path(args.work_dir)
    parts = []
    for subset in args.subsets:
        exp_id = f"{args.key}{'_control' if control else ''}__{subset}"
        pkl = work / f"trial{exp_id}_samples_{lang}.pkl"
        mbbq(work, "generate_samples", lang, exp_id, control, [subset])
        if "answer" not in load(pkl).columns:
            if model_path is None:
                sys.exit(f"[{tag}] {subset} needs ask_model but no model given")
            info.setdefault("gpu", check_gpu())
            t0 = time.time()
            mbbq(work, "ask_model", lang, exp_id, control, [subset],
                 model=model_path, bs=args.bs)
            secs = round(time.time() - t0)
            info["ask_seconds"][f"{tag}/{subset}"] = secs
            print(f"[{tag}] {subset}: ask_model took {secs / 3600:.2f} h")
        # mbbq.py's detect_answers "succeeds" on a pickle with no answers,
        # so check before detecting.
        if "answer" not in load(pkl).columns:
            sys.exit(f"[{tag}] {subset}: no answer column after ask_model")
        mbbq(work, "detect_answers", lang, exp_id, control, [subset])
        parts.append(load(pkl))

    import pandas as pd
    merged = pd.concat(parts, ignore_index=True)
    ref_id = f"{args.key}{'_control' if control else ''}__reference"
    ref_pkl = work / f"trial{ref_id}_samples_{lang}.pkl"
    mbbq(work, "generate_samples", lang, ref_id, control, args.subsets)
    ref = load(ref_pkl)
    ref_pkl.unlink()
    pd.testing.assert_frame_equal(merged[KEY_COLS], ref[KEY_COLS])
    assert merged["answer"].notna().all(), "missing model answers"
    assert "answer_detected" in merged.columns, "detect_answers did not run"

    undetected = float(merged["answer_detected"].isna().mean())
    dump_atomic(merged, out)
    info["results"][tag] = dict(rows=len(merged), undetected=round(undetected, 4),
                                subsets=args.subsets,
                                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[{tag}] wrote {out} ({len(merged)} rows, {undetected:.1%} undetected)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--key", required=True, help="short model name, e.g. mistral")
    ap.add_argument("--hf_model", help="HF repo id; omit if all answers exist")
    ap.add_argument("--revision", help="model commit (default: current main)")
    ap.add_argument("--langs", nargs="+", default=LANGS)
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    ap.add_argument("--no_control", action="store_true", help="main set only")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    args = ap.parse_args()

    work, results = Path(args.work_dir), Path(args.results_dir)
    work.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    if not (work / "data").exists():
        (work / "data").symlink_to(ROOT / "data")

    info_path = results / f"{args.key}_runinfo.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info.setdefault("results", {})
    info.setdefault("ask_seconds", {})

    model_path = None
    if args.hf_model:
        import torch
        import transformers
        sha, model_path = resolve_model(args.hf_model, args.revision, info)
        info.update(hf_model=args.hf_model, revision=sha, bs=args.bs,
                    transformers=transformers.__version__,
                    torch=torch.__version__, python=platform.python_version())
        print(f"{args.hf_model} @ {sha}")

    try:
        for lang in args.langs:
            for control in [False] if args.no_control else [False, True]:
                run_one(args, lang, control, model_path, info)
    finally:
        info_path.write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
