# my_research

Russian extension of MBBQ. Upstream files are untouched; all work lives here.

- `scorer/`: scorer (`mbbq_scorer.py`, the frozen method below), unit tests,
  Table 3 check (`reproduce_table3.py`: scores en, nl, es and tr against the
  paper and prints each deviation and the largest ones)
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

## Scoring method

Frozen on 2026-10-13. `scorer/mbbq_scorer.py` implements only this method; use
it for every score from now on, including Russian.

| Choice | Setting |
|---|---|
| Biased answer | The target, whatever the polarity (upstream `score_answer`) |
| Template filter | Keep templates with above-chance disambiguated control accuracy in all languages |
| Bias_D | Divided by the number of disambiguated contexts (Eq. 2) |
| Averaging | Score each template (prompt-averaged), then average templates equally |

Found on Mistral's Table 3 and checked on Zephyr without retuning.
- **Largest deviations from Table 3**, over all languages and metrics:
  - accuracies within 1.34 points;
  - bias scores within 0.0136;
  - Mistral alone: 0.96 points and 0.0035.
- **Templates kept:** 57 vs. 56 (Mistral), 54 vs. 58 (Zephyr). Zephyr's count is
  unstable near the 1/3 cutoff in Turkish.

Details: `scorer/SCORING_CONDITIONS.md`.

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

Status: all 8 subsets done (en, nl, es, tr, each main + control; 2026-10-10).
Table 3 gate run 2026-10-10: all 4 languages within tolerance under the
scoring method above; gate still FAILs on #T (57 vs. the paper's 56). Method
frozen 2026-10-13 after Zephyr (above).
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

### Zephyr (Oct 2026, Colab, `MyDrive/mbbq_paper`)

Status: all 8 subsets done (en, nl, es, tr, each main + control; 2026-10-13).
- **Table 3 gate run 2026-10-13:** strict FAIL.
  - Under the frozen method, en is within ±2 / ±0.01.
  - nl Bias_A, es Bias_D and tr Bias_A are 0.0107–0.0136 off.
  - #T is 54 vs. the paper's 58.
  - All accuracies are within 1.34 points.
- **Settings:** every output is generated with the settings above:
  `HuggingFaceH4/zephyr-7b-beta` at the pinned commit, `max_new_tokens=256`,
  batch size 32. `run_step6.py` refuses to resume under other settings.
- **GPU:** NVIDIA A100-SXM4-40GB in the logged session (nl, nl_control); not
  checked for the other sessions.

| | Value |
|---|---|
| `transformers` | 4.39.3 (pinned by the notebook) |
| `torch` | 2.11.0+cu130, Colab's default (see Mistral) |
| Python | 3.13 |

Answers at the 256-token cap: en 1.4–1.6%, es 3.0–3.5%, nl 7.3–8.7%, tr
25.6–31.2%. The answer usually comes first, so 96–98% of capped answers are
still detected. Undetected overall: 0.46–2.07% per file.
