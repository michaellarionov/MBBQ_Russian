"""Ask a model the MBBQ questions with the paper-era (April 2024) settings.

Upstream mbbq.py (Dec 2024) uses one setup for every model: chat template and
max_new_tokens=100. The code the authors released with the paper (models.py at
commit 20ae39b, 2024-04-09; view with `git show 20ae39b:models.py`) set prompt
format, generation length and batch size per model. This replays it:

  Mistral: "<s>[INST] {q} [/INST]",                   1000 new tokens, bs 16
  Zephyr:  "<|user|>\\n{q}</s>\\n<|assistant|>\\n",    256 new tokens, bs 32

Both: greedy (do_sample=False, num_beams=1), bf16, full text returned and the
answer split off after the prompt, as models.py did. The tokenizer adds its own
BOS, so Mistral prompts start with two BOS tokens, also as in models.py. Model
revisions are the last HF commits before the March 2024 runs.

REPLICATION_MODELS are not from the paper: newer models for the gender-leak
replication (DECISIONS.md, "Replication on two more models"). Each prompt is
the model's own chat template for one user turn; decoding is greedy, as above,
with repetition_penalty=1.0 (both ship 1.05) and left padding (the Qwen2
tokenizer has no padding side of its own in transformers 4.39.3, and right
padding breaks batched generation). They need transformers 4.46.3: their
tokenizer.json files were saved by a newer `tokenizers` than 4.39.3's.

FIFTH_MODELS are the candidates for a fifth model (DECISIONS.md, "Fifth
model"), with the paper-era settings from models.py at 20ae39b:
  Aya-101 (first choice): a 13B mT5 encoder-decoder, "text2text-generation",
    the bare question (no prompt format), 128 new tokens, bs 8. The pipeline
    returns only the generated text, so there is nothing to split off.
    models.py passed no decoding flags; Aya's generation_config sets none, so
    it was greedy, which do_sample=False, num_beams=1 makes explicit.
  Llama-2-7B-Chat (fallback): "<s>[INST] {q} [/INST]", bs 4, greedy, split
    on "[/INST]". models.py set no max_new_tokens, so generation stops at
    EOS or the model's generation_config max_length; max_new_tokens=None
    reproduces that. Gated on HF (manual approval by Meta).
"""
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

PAPER_MODELS = {
    "mistral": dict(
        hf_model="mistralai/Mistral-7B-Instruct-v0.2",
        revision="41b61a33a2483885c981aa79e0df6b32407ed873",  # 2024-03-24
        prompt="<s>[INST] {} [/INST]",
        max_new_tokens=1000,
        bs=16,
        # models.py tried "[/INST] " first, then "[/INST]"
        splits=["[/INST] ", "[/INST]"],
    ),
    "zephyr": dict(
        hf_model="HuggingFaceH4/zephyr-7b-beta",
        revision="b70e0c9a2d9e14bd1e812d3c398e5f313e93b473",  # 2024-02-29
        prompt="<|user|>\n{}</s>\n<|assistant|>\n",
        max_new_tokens=256,
        bs=32,
        splits=["<|assistant|>\n"],
    ),
}

REPLICATION_MODELS = {
    "qwen": dict(
        hf_model="Qwen/Qwen2.5-7B-Instruct",
        revision="a09a35458c702b33eeacc393d103063234e8bc28",  # 2025-01-12
        # the template adds Qwen's default system prompt when none is given
        prompt="<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a "
               "helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
               "<|im_start|>assistant\n",
        max_new_tokens=512,
        bs=16,
        splits=["<|im_start|>assistant\n"],
        padding_side="left",
        generate_kwargs=dict(repetition_penalty=1.0),
        transformers="4.46.3",  # 4.39.3's tokenizers can't read its tokenizer.json
    ),
    "tlite": dict(
        hf_model="t-tech/T-lite-it-1.0",
        revision="fbabc76f32140416dc5b0ceef392c7778eec1312",  # 2024-12-13
        prompt="<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        max_new_tokens=512,
        bs=16,
        splits=["<|im_start|>assistant\n"],
        padding_side="left",
        generate_kwargs=dict(repetition_penalty=1.0),
        transformers="4.46.3",  # 4.39.3's tokenizers can't read its tokenizer.json
    ),
}
FIFTH_MODELS = {
    "aya": dict(
        hf_model="CohereLabs/aya-101",  # formerly CohereForAI/aya-101 (redirects)
        # last commit before the March 2024 runs; weights unchanged since 2024-02-13
        revision="bb36885b37c25bb1a5f621d324dc861ab92e8ae5",  # 2024-02-20
        task="text2text-generation",
        prompt="{}",
        max_new_tokens=128,
        bs=8,
        splits=None,
    ),
    "llama2": dict(
        hf_model="meta-llama/Llama-2-7b-chat-hf",
        # HEAD as of 2026-11-08 (last modified 2024-04-17); the commit list is
        # gated, so pin the last pre-March-2024 commit once access is granted.
        revision="f5db02db724555f92da89c216ac04704f23d4590",
        prompt="<s>[INST] {} [/INST]",
        max_new_tokens=None,
        bs=4,
        splits=["[/INST]"],
    ),
}
MODELS = {**PAPER_MODELS, **REPLICATION_MODELS, **FIFTH_MODELS}


def load_model(model_path, padding_side=None, task="text-generation"):
    pipe = pipeline(task, model=model_path,
                    torch_dtype=torch.bfloat16, device_map="auto")
    if padding_side:  # REPLICATION_MODELS; the paper models keep the tokenizer's
        pipe.tokenizer.padding_side = padding_side
    # Mistral has no pad token (models.py set it to EOS); Zephyr's is EOS.
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
    return pipe


def split_answer(text, splits):
    if splits is None:  # text2text-generation returns only the answer
        return text
    for s in splits[:-1]:
        if s in text:
            return text.split(s)[1]
    return text.split(splits[-1])[1]


def ask(pipe, cfg, questions, bs):
    """Model answers to `questions` (list of str), in order."""
    # A plain column, not dataset.map: mapping a function that closes over the
    # pipeline makes datasets pickle the whole model to fingerprint it.
    ds = Dataset.from_dict({"question": [cfg["prompt"].format(q) for q in questions]})
    # Llama-2: no max_new_tokens at all, as in models.py (None would still be
    # passed on to generate)
    cap = {} if cfg["max_new_tokens"] is None else {"max_new_tokens": cfg["max_new_tokens"]}
    out = pipe(KeyDataset(ds, "question"), batch_size=bs, **cap, do_sample=False, num_beams=1,
               # what generate() falls back to anyway; passing it avoids a
               # warning per batch
               pad_token_id=pipe.tokenizer.pad_token_id, **cfg.get("generate_kwargs", {}))
    return [split_answer(r[0]["generated_text"], cfg["splits"])
            for r in tqdm(out, total=len(ds))]
