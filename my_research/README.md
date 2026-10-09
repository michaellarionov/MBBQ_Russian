# my_research

Russian extension of MBBQ. Upstream files are untouched; all work lives here.

- `scorer/`: scorer (`mbbq_scorer.py`), unit tests, Table 3 reproduction sweep
  (`reproduce_table3.py`: ranks the switch combinations by English error, then
  requires nl, es and tr within tolerance under the same combination)
- `generation/`: step 6, producing model outputs (run `colab_step6.ipynb` on an A100)
  - `ask_model_paper.py` asks the model with the paper-era settings from the
    authors' `models.py` (commit `20ae39b`, 2024-04-09, deleted upstream in
    Dec 2024; `git show 20ae39b:models.py`), not upstream `ask_model`, which
    uses 100 new tokens for every model. See the table below.
  - `run_mbbq.py` runs upstream `mbbq.py` unchanged (used here for
    `generate_samples` and `detect_answers`) but supplies the `torch` and
    `KeyDataset` imports it lacks (as shipped, `-mode ask_model` raises NameError)
    and skips `login()` when no `-token` is given. It also stops `datasets` from
    pickling the whole model to fingerprint `ask_model`'s `map` call, which used
    tens of GB of RAM and got the process SIGKILLed on Colab.
  - `run_step6.py` runs one subset at a time and resumes after disconnects. It
    loads the pinned model once, checks the merged rows against a full
    `generate_samples`, and won't save outputs without answers. It stamps the
    generation settings into its work folder (`<key>_settings.json`) and stops
    rather than reuse files made with other settings.
- `templates/`: Russian templates and slot lexicon
- `results/`: renamed `mbbq.py` pickles, `<model>_<lang>[_control].pkl`

## Frozen scorer switches

Not frozen yet: the Table 3 gate has not been run on real model outputs.

| Switch | Value |
|---|---|
| `filter_on` | TBD |
| `bias_d` | TBD |
| `average` | TBD |

## Generation settings

From `models.py@20ae39b`; set in `generation/ask_model_paper.py`. Both models:
greedy (`do_sample=False`, 1 beam), bf16, no system prompt, `transformers==4.39.3`.
The pinned revisions' chat templates render exactly these prompt strings.

| | Mistral-7B-Instruct-v0.2 | zephyr-7b-beta |
|---|---|---|
| Commit | `41b61a33a2483885c981aa79e0df6b32407ed873` (2024-03-24, last before the March 2024 runs) | `b70e0c9a2d9e14bd1e812d3c398e5f313e93b473` (2024-02-29) |
| Prompt | `<s>[INST] {q} [/INST]` | `<\|user\|>\n{q}</s>\n<\|assistant\|>\n` |
| `max_new_tokens` | 1000 | 256 |
| Batch size | 16 | 32 |

## Run record

### Mistral (Oct 2026, Colab, `MyDrive/mbbq_paper`)

Status: en, en_control, nl, nl_control done; es and tr running (2026-10-09).
Every output is generated with the settings above: `mistralai/Mistral-7B-Instruct-v0.2` at the pinned commit,
`max_new_tokens=1000`, batch size 16, on an NVIDIA A100-SXM4-40GB in every
session. `run_step6.py` refuses to resume under other settings, so no output
mixes settings.

| | Value |
|---|---|
| `transformers` | 4.39.3 (pinned by the notebook) |
| `torch` | 2.11.0+cu130, Colab's default, not the 2.2.2 in `requirements.txt`; may cause small numerical differences under greedy decoding |
| Python | 3.13 |

Per-file row counts and undetected rates come from the pickles.
