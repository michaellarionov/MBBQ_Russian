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


def load_model(model_path):
    pipe = pipeline("text-generation", model=model_path,
                    torch_dtype=torch.bfloat16, device_map="auto")
    # Mistral has no pad token (models.py set it to EOS); Zephyr's is EOS.
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
    return pipe


def split_answer(text, splits):
    for s in splits[:-1]:
        if s in text:
            return text.split(s)[1]
    return text.split(splits[-1])[1]


def ask(pipe, cfg, questions, bs):
    """Model answers to `questions` (list of str), in order."""
    # A plain column, not dataset.map: mapping a function that closes over the
    # pipeline makes datasets pickle the whole model to fingerprint it.
    ds = Dataset.from_dict({"question": [cfg["prompt"].format(q) for q in questions]})
    out = pipe(KeyDataset(ds, "question"), batch_size=bs,
               max_new_tokens=cfg["max_new_tokens"], do_sample=False, num_beams=1,
               # what generate() falls back to anyway; passing it avoids a
               # warning per batch
               pad_token_id=pipe.tokenizer.pad_token_id)
    return [split_answer(r[0]["generated_text"], cfg["splits"])
            for r in tqdm(out, total=len(ds))]
