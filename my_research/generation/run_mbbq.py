"""Run upstream mbbq.py unchanged, adding the two imports it is missing.

mbbq.py uses torch.bfloat16 and KeyDataset in ask_model but imports neither,
so `-mode ask_model` raises NameError. It also calls login(token) even when
-token is empty, which crashes; here an empty -token skips login and the
cached HF login / HF_TOKEN env var is used. Arguments pass straight through:
  python my_research/generation/run_mbbq.py -mode ask_model -lang en ...
Like mbbq.py, it reads data/ and writes trial<exp_id>_samples_<lang>.pkl
relative to the current directory.
"""
import runpy
import sys
from pathlib import Path

import huggingface_hub
import torch
from transformers.pipelines.pt_utils import KeyDataset

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True  # keep the upstream tree clean
sys.path.insert(0, str(ROOT))  # for mbbq.py's `from answer_detection import *`
_login = huggingface_hub.login
huggingface_hub.login = lambda token=None, *a, **k: _login(token, *a, **k) if token else None

runpy.run_path(str(ROOT / "mbbq.py"), run_name="__main__",
               init_globals={"torch": torch, "KeyDataset": KeyDataset})
