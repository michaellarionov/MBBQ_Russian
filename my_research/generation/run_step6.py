"""Step 6: generate MBBQ model outputs, resumably (built for Colab).

For each language x {main, control} x subset: upstream mbbq.py (through
run_mbbq.py) runs generate_samples in --work_dir, ask_model_paper.py asks the
model with the paper-era settings (per-model prompt format, max_new_tokens,
batch size; see that file), and mbbq.py runs detect_answers. A dropped session
loses at most one subset; steps finished earlier in the same run are skipped on
rerun. get_samples is deterministic, so the per-subset pickles concatenate to
the same rows as one all-subsets run; this is checked against a fresh
generate_samples before anything is written to --results_dir.

The generation settings are stamped into --work_dir and --results_dir. A rerun
resumes only under identical settings; files from a run with other settings
(e.g. upstream ask_model's 100 tokens) stop the script instead of being reused.

Output: <results_dir>/<key>_<lang>[_control].pkl (what reproduce_table3.py
reads) and <key>_runinfo.json (settings, versions, timings).

Log in to Hugging Face first (huggingface-cli login or HF_TOKEN).

  python my_research/generation/run_step6.py --model mistral \
      --work_dir /content/drive/MyDrive/mbbq_paper/work \
      --results_dir /content/drive/MyDrive/mbbq_paper/results
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
# Copied from ask_model_paper.PAPER_MODELS so --help works without torch;
# main() checks they agree.
MODELS = ["mistral", "zephyr"]
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


def mbbq(work_dir, mode, lang, exp_id, control, subsets):
    cmd = [sys.executable, str(RUN_MBBQ), "-mode", mode, "-lang", lang,
           "-exp_id", exp_id, "-subsets", *subsets]
    if control:
        cmd.append("--control")
    subprocess.run(cmd, cwd=work_dir, check=True)


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU: generation needs one (Colab: A100 or L4 runtime).")
    # is_bf16_supported() is True on a T4 too (emulated, very slow), so check
    # for Ampere or newer, which has native bf16.
    if torch.cuda.get_device_capability()[0] < 8:
        sys.exit(f"{torch.cuda.get_device_name()} has no native bf16 (T4?); "
                 "generation runs in bf16. Switch to an A100 or L4 runtime.")
    return torch.cuda.get_device_name()


def check_settings(work, key, settings, info):
    """Refuse to resume from files written under other generation settings."""
    if info and info.get("settings") != settings:
        sys.exit(f"{key}_runinfo.json in --results_dir is from a run with other "
                 f"generation settings ({info.get('settings', 'none recorded')}). "
                 "Use a fresh --results_dir or delete the old outputs.")
    stamp = work / f"{key}_settings.json"
    if stamp.exists():
        if json.loads(stamp.read_text()) != settings:
            sys.exit(f"{stamp} is from a run with other generation settings. "
                     "Use a fresh --work_dir or delete its trial*.pkl files.")
        return
    old = [*work.glob(f"trial{key}__*.pkl"), *work.glob(f"trial{key}_control__*.pkl")]
    if old:
        sys.exit(f"--work_dir has {len(old)} {key} pickles from an earlier run with "
                 f"unknown settings (e.g. {old[0].name}). Use a fresh --work_dir "
                 "or delete them.")
    stamp.write_text(json.dumps(settings, indent=2))


class Asker:
    """Loads the pinned model on first use, then keeps it for every subset."""

    def __init__(self, cfg, bs, info):
        self.cfg, self.bs, self.info, self.pipe = cfg, bs, info, None

    def __call__(self, questions):
        import ask_model_paper
        if self.pipe is None:
            from huggingface_hub import snapshot_download
            self.info.setdefault("gpu", check_gpu())
            path = snapshot_download(self.cfg["hf_model"], revision=self.cfg["revision"],
                                     allow_patterns=WEIGHTS)
            self.pipe = ask_model_paper.load_model(path)
        return ask_model_paper.ask(self.pipe, self.cfg, questions, self.bs)


def run_one(args, lang, control, asker, info):
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
        df = load(pkl)
        if "answer" not in df.columns:
            t0 = time.time()
            df["answer"] = asker(df["question"].tolist())
            secs = round(time.time() - t0)
            dump_atomic(df, pkl)
            info["ask_seconds"][f"{tag}/{subset}"] = secs
            print(f"[{tag}] {subset}: generation took {secs / 3600:.2f} h")
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
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--key", help="name for output files (default: --model)")
    ap.add_argument("--langs", nargs="+", default=LANGS)
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    ap.add_argument("--no_control", action="store_true", help="main set only")
    ap.add_argument("--bs", type=int,
                    help="batch size (default: the paper's, per model; changing "
                         "it changes outputs slightly)")
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    args = ap.parse_args()
    args.key = args.key or args.model

    import torch
    import transformers
    from ask_model_paper import PAPER_MODELS
    assert sorted(PAPER_MODELS) == sorted(MODELS), "update MODELS"
    cfg = PAPER_MODELS[args.model]
    bs = args.bs or cfg["bs"]
    settings = dict(model=args.model, hf_model=cfg["hf_model"], revision=cfg["revision"],
                    prompt=cfg["prompt"], max_new_tokens=cfg["max_new_tokens"], bs=bs)

    work, results = Path(args.work_dir), Path(args.results_dir)
    work.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    if not (work / "data").exists():
        (work / "data").symlink_to(ROOT / "data")

    info_path = results / f"{args.key}_runinfo.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    check_settings(work, args.key, settings, info)
    info.update(settings=settings, transformers=transformers.__version__,
                torch=torch.__version__, python=platform.python_version())
    info.setdefault("results", {})
    info.setdefault("ask_seconds", {})
    print(f"{cfg['hf_model']} @ {cfg['revision']}: "
          f"max_new_tokens={cfg['max_new_tokens']}, bs={bs}")

    asker = Asker(cfg, bs, info)
    try:
        for lang in args.langs:
            for control in [False] if args.no_control else [False, True]:
                run_one(args, lang, control, asker, info)
    finally:
        info_path.write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
