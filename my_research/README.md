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
    tens of GB of RAM and got the process SIGKILLed on Colab. For `-lang ru`
    and the leak sets `ru_masc` and `ru_fem` it registers the Russian prompts
    and detection phrases from `detection/` at runtime; other languages run
    exactly as upstream. Since
    upstream `7999afd` (2026-10-15) points `target_loc` at the other person on
    non-negative questions, it restores the published `target_loc` (the
    target group's answer on every question), so new runs match the stored
    ones; the scorer gives sign-corrected scores separately
    (`annotate(..., sign="corrected")`, "Bias sign" below).
  - `run_step6.py` runs one subset at a time and resumes after disconnects. It
    loads the pinned model once, checks the merged rows against a full
    `generate_samples`, and won't save outputs without answers. It stamps the
    generation settings into its work folder (`<key>_settings.json`) and stops
    rather than reuse files made with other settings.
- `templates/`: Russian templates and slot lexicon; `generate_ru.py` writes
  `data/*_ru.jsonl`, `check_ru.py` runs every mechanical check (run it after
  each edit)
- `detection/`: Russian prompts and answer-detection phrases
  (`ru_detection.py`) and response normalisation (`normalize_letters.py`:
  ё → е, Cyrillic А/С option letters → Latin), used through `run_mbbq.py`
- `results/`: renamed `mbbq.py` pickles, `<model>_<lang>[_control].pkl`

## Scoring method

Frozen on 2026-10-15. `scorer/mbbq_scorer.py` implements only this method; use
it for every score from now on, including Russian. It follows the MBBQ first
author's description of the paper's scoring (email, 2026-10-15).

| Choice | Setting |
|---|---|
| Denominators | Only rows with a detected answer, in every score and in the template filter |
| Biased answer | The target, whatever the polarity (upstream `score_answer`; the author confirms Table 3 uses it) |
| Template filter | Per model: keep templates with above-chance disambiguated control accuracy in all languages |
| Bias_D | Divided by the number of disambiguated contexts (Eq. 2) |
| Averaging | Score each template on its rows pooled across the prompts, then average templates equally |

- **Largest deviations from Table 3**, over all languages and metrics:
  - accuracies within 1.13 points;
  - bias scores within 0.0173;
  - Mistral alone: 0.95 points and 0.0035.
- **Templates kept:** 57 vs. 56 (Mistral), 56 vs. 58 (Zephyr). Zephyr's count is
  unstable near the 1/3 cutoff.

Details: `scorer/SCORING_CONDITIONS.md`.

### Bias sign

`annotate(df, sign=...)` picks the biased answer; nothing else in the method
changes with it.

| `sign` | Biased answer | Use |
|---|---|---|
| `published` (default) | The target group, whatever the polarity (released code) | Table 3 reproduction; gives exactly the frozen numbers above |
| `corrected` | The target on a negative question, the other person on a non-negative one (BBQ's definition; upstream `7999afd`) | New claims |

The sign changes Bias_A and the Bias_D context type (biased vs.
counter-biased), never accuracy, the undetected counts or the template filter.
`corrected` needs `question_polarity` (`attach_polarity`).
`reproduce_table3.py` uses the default, `published`.

`scorer/score_both_signs.py` scores every existing output under both signs
and writes `results/sign/`:
- `scores_published.csv`, `scores_corrected.csv`: one row per model ×
  config × lang × set × metric, accuracies as fractions. `n_rows` and
  `n_undetected` count the context condition's rows (ambiguous for `_A`,
  disambiguated for `_D`). Configs:
  - A: Table 3, en/nl/es/tr on the frozen filter;
  - B: en/nl/es/tr/ru and ru control, all 98 templates;
  - C: ru vs. ru_masc on paired items.

`scorer/fem_analysis.py` compares feminine and masculine question agreement
on Gender_identity (`ru`, `ru_masc`, `ru_fem`, and the `ru_masc` rerun as the
noise floor), with the tests fixed in `docs/DECISIONS.md` ("Feminine
variant") before the run, and writes `results/fem/`.
- `per_template_published.csv`, `per_template_corrected.csv`: the same
  metrics per template, `n_rows` = detected rows behind the value. Use these
  for significance tests (templates as the unit).
- `SIGN_COMPARISON.md`: both signs side by side, with language rankings.

Each file name gives its sign. Russian answers are re-detected with the
current detection code; other languages keep their stored detection. The
script stops if accuracy, the undetected counts or the filter differ between
the signs.

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
Under the scoring method above, all 4 languages are within ±2 / ±0.01 of
Table 3; #T is 57 vs. the paper's 56.
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
- **Table 3, under the scoring method above:** nl and tr are within ±2 / ±0.01.
  - en Bias_D and es Bias_D are 0.0107 and 0.0173 off.
  - #T is 56 vs. the paper's 58.
  - All accuracies are within 1.13 points.
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
