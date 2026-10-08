# my_research

Russian extension of MBBQ. Upstream files are untouched; all work lives here.

- `scorer/`: scorer (`mbbq_scorer.py`), unit tests, Table 3 reproduction sweep
- `generation/`: step 6, producing model outputs (run `colab_step6.ipynb` on an A100)
  - `run_mbbq.py` runs upstream `mbbq.py` unchanged but supplies the `torch` and
    `KeyDataset` imports it lacks (as shipped, `-mode ask_model` raises NameError)
    and skips `login()` when no `-token` is given. It also stops `datasets` from
    pickling the whole model to fingerprint `ask_model`'s `map` call, which used
    tens of GB of RAM and got the process SIGKILLed on Colab.
  - `run_step6.py` runs one subset at a time and resumes after disconnects. It
    pins the model commit, checks the merged rows against a full
    `generate_samples`, and won't save outputs without answers.
- `templates/`: Russian templates and slot lexicon
- `results/`: renamed `mbbq.py` pickles, `<model>_<lang>[_control].pkl`

## Frozen scorer switches

Not frozen yet: the Table 3 gate has not been run on real model outputs.

| Switch | Value |
|---|---|
| `filter_on` | TBD |
| `bias_d` | TBD |
| `average` | TBD |
| Mistral-7B-Instruct-v0.2 commit | `41b61a33a2483885c981aa79e0df6b32407ed873` (2024-03-24, last before the paper) |
| zephyr-7b-beta commit | `b70e0c9a2d9e14bd1e812d3c398e5f313e93b473` (2024-02-29) |
