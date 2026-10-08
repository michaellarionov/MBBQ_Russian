"""Run upstream mbbq.py unchanged, working around three bugs in it.

mbbq.py uses torch.bfloat16 and KeyDataset in ask_model but imports neither,
so `-mode ask_model` raises NameError. It also calls login(token) even when
-token is empty, which crashes; here an empty -token skips login and the
cached HF login / HF_TOKEN env var is used. And ask_model runs out of RAM
on a 7B model (see the datasets patch below). Arguments pass straight through:
  python my_research/generation/run_mbbq.py -mode ask_model -lang en ...
Like mbbq.py, it reads data/ and writes trial<exp_id>_samples_<lang>.pkl
relative to the current directory.
"""
import runpy
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
_login = huggingface_hub.login
huggingface_hub.login = lambda token=None, *a, **k: _login(token, *a, **k) if token else None

runpy.run_path(str(ROOT / "mbbq.py"), run_name="__main__",
               init_globals={"torch": torch, "KeyDataset": KeyDataset})
