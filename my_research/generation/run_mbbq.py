"""Run upstream mbbq.py unchanged, working around three bugs in it and adding
Russian.

mbbq.py uses torch.bfloat16 and KeyDataset in ask_model but imports neither,
so `-mode ask_model` raises NameError. It also calls login(token) even when
-token is empty, which crashes; here an empty -token skips login and the
cached HF login / HF_TOKEN env var is used. And ask_model runs out of RAM
on a 7B model (see the datasets patch below). Arguments pass straight through:
  python my_research/generation/run_mbbq.py -mode ask_model -lang en ...

Russian (-lang ru, and ru_masc for Gender_identity_ru_masc.jsonl): mbbq.py
hard-codes its languages, so the Russian prompts and detection phrases
(my_research/detection/ru_detection.py) are added to answer_detection's
dictionaries before mbbq.py imports them, -lang accepts ru and ru_masc, and
detect_answer normalises Russian responses first (normalize_letters.py).
Other languages run exactly as upstream.

target_loc: upstream 7999afd (2026-10-15) points it at the other person on
non-negative questions. Every MBBQ run here, and the published scores, use the
earlier definition (the target group's answer on every question), so when
mbbq.py has that change, get_samples is wrapped to restore it
(mbbq_scorer.flip_target_loc). The scorer computes sign-corrected scores
separately.
Like mbbq.py, it reads data/ and writes trial<exp_id>_samples_<lang>.pkl
relative to the current directory.
"""
import argparse
import ast
import sys
from pathlib import Path

import datasets
import datasets.fingerprint
import huggingface_hub
import torch
import transformers
from transformers.pipelines.pt_utils import KeyDataset

# generate() logs "Setting `pad_token_id` to `eos_token_id`" once per batch
# (~25k lines per model), which bloats the Colab output. It is only a notice:
# that padding is what happens either way.
transformers.logging.set_verbosity_error()

# ask_model's dataset.map(lambda x: process(model, x)) makes datasets
# fingerprint the lambda by pickling it into memory, closure included: the
# whole pipeline, weights and all (~2x the weights, then copied again). For a
# 7B model that is tens of GB of RAM and Colab SIGKILLs the process. The
# dataset is in-memory (from_pandas), so nothing is cached and a random
# fingerprint changes nothing else. Modules import update_fingerprint by
# name (arrow_dataset does), so replace every reference.
_orig_update = datasets.fingerprint.update_fingerprint
for _mod in list(sys.modules.values()):
    if (getattr(_mod, "__name__", "").startswith("datasets")
            and getattr(_mod, "update_fingerprint", None) is _orig_update):
        _mod.update_fingerprint = (
            lambda *a, **k: datasets.fingerprint.generate_random_fingerprint())

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True  # keep the upstream tree clean
sys.path.insert(0, str(ROOT))  # for mbbq.py's `from answer_detection import *`
sys.path.insert(0, str(ROOT / "my_research" / "detection"))
sys.path.insert(0, str(ROOT / "my_research" / "scorer"))
_login = huggingface_hub.login
huggingface_hub.login = lambda token=None, *a, **k: _login(token, *a, **k) if token else None

import answer_detection  # noqa: E402  (upstream; mbbq.py's star import reuses this module)
import ru_detection  # noqa: E402
from mbbq_scorer import attach_polarity, flip_target_loc  # noqa: E402

ru_detection.register(answer_detection)
_add_argument = argparse.ArgumentParser.add_argument


def _add_lang_choices(self, *names, **kw):
    if names == ("-lang",):
        kw["choices"] = [*kw["choices"], *ru_detection.LANGS]
    return _add_argument(self, *names, **kw)


argparse.ArgumentParser.add_argument = _add_lang_choices

# Run mbbq.py's definitions, wrap detect_answer and get_samples, then run its
# __main__ block:
# detect_answers looks detect_answer up in the module globals at call time.
MBBQ = ROOT / "mbbq.py"
*defs, main = ast.parse(MBBQ.read_text(), str(MBBQ)).body
assert isinstance(main, ast.If) and ast.unparse(main.test) == "__name__ == '__main__'"
ns = {"__name__": "__main__", "__file__": str(MBBQ), "torch": torch, "KeyDataset": KeyDataset}
exec(compile(ast.Module(defs, []), str(MBBQ), "exec"), ns)
ns["detect_answer"] = ru_detection.wrap_detect_answer(ns["detect_answer"])


def published_target_loc(get_samples):
    def wrapped(subsets, control=False, language="en", **kw):
        df = get_samples(subsets, control=control, language=language, **kw)
        # get_samples reads data/ relative to the current directory too
        pol = attach_polarity(df, "data", language, control)["question_polarity"]
        df["target_loc"] = flip_target_loc(df["target_loc"], df["unknown_loc"],
                                           pol.set_axis(df.index))
        return df
    return wrapped


(get_samples_def,) = [n for n in defs if isinstance(n, ast.FunctionDef) and n.name == "get_samples"]
if "other_loc" in ast.unparse(get_samples_def):  # upstream 7999afd or later
    ns["get_samples"] = published_target_loc(ns["get_samples"])
exec(compile(ast.Module(main.body, []), str(MBBQ), "exec"), ns)
