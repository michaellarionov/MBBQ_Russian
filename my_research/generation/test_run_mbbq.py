import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN_MBBQ = Path(__file__).resolve().parent / "run_mbbq.py"


def published_target_loc(jsonl):
    """The target group's answer on every question, whatever its polarity."""
    out = []
    for line in open(jsonl):
        s = json.loads(line)
        groups = s["additional_metadata"]["stereotyped_groups"]
        locs = [int(k[-1]) for k, v in s["answer_info"].items() if v[1] in groups]
        out += [locs[0] if len(locs) == 1 else 100] * 5
    return out


@pytest.mark.parametrize("control", [False, True])
def test_generate_samples_gives_published_target_loc(tmp_path, control):
    (tmp_path / "data").symlink_to(ROOT / "data")
    cmd = [sys.executable, str(RUN_MBBQ), "-mode", "generate_samples", "-lang", "en",
           "-exp_id", "t", "-subsets", "Age"] + (["--control"] if control else [])
    subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True)
    df = pd.read_pickle(tmp_path / "trialt_samples_en.pkl")
    jsonl = ROOT / "data" / f"Age{'_control' if control else ''}_en.jsonl"
    pol = [json.loads(line)["question_polarity"] for line in open(jsonl) for _ in range(5)]
    assert "nonneg" in pol
    assert list(df["target_loc"]) == published_target_loc(jsonl)
