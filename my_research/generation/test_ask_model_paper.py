import pytest

from ask_model_paper import PAPER_MODELS, REPLICATION_MODELS, ask, split_answer
from run_step6 import settings_for


class FakeTokenizer:
    pad_token_id = 2


class FakePipe:
    """Echoes the prompt plus a canned continuation, like return_full_text=True."""
    tokenizer = FakeTokenizer()

    def __init__(self, continuation):
        self.continuation, self.calls = continuation, []

    def __call__(self, prompts, **kw):
        self.calls.append(kw)
        return ([{"generated_text": p + self.continuation}] for p in prompts)


def test_paper_token_limits_and_batch_sizes():
    # models.py at commit 20ae39b
    assert PAPER_MODELS["mistral"]["max_new_tokens"] == 1000
    assert PAPER_MODELS["mistral"]["bs"] == 16
    assert PAPER_MODELS["zephyr"]["max_new_tokens"] == 256
    assert PAPER_MODELS["zephyr"]["bs"] == 32


def test_mistral_prompt_split_and_generation_kwargs():
    pipe = FakePipe(" (b) The grandson.")
    cfg = PAPER_MODELS["mistral"]
    assert ask(pipe, cfg, ["Q1", "Q2"], bs=16) == ["(b) The grandson."] * 2
    kw = pipe.calls[0]
    assert kw["max_new_tokens"] == 1000 and kw["batch_size"] == 16
    assert kw["do_sample"] is False and kw["num_beams"] == 1


def test_mistral_split_falls_back_without_space():
    # models.py: split on "[/INST] ", and on "[/INST]" if that fails
    cfg = PAPER_MODELS["mistral"]
    assert split_answer("<s>[INST] Q [/INST]\n(a)", cfg["splits"]) == "\n(a)"


def test_zephyr_prompt_and_split():
    pipe = FakePipe("(c) Unknown")
    cfg = PAPER_MODELS["zephyr"]
    assert ask(pipe, cfg, ["Q"], bs=32) == ["(c) Unknown"]
    assert pipe.calls[0]["max_new_tokens"] == 256
    assert cfg["prompt"].format("Q") == "<|user|>\nQ</s>\n<|assistant|>\n"


def test_braces_in_question_survive_formatting():
    assert PAPER_MODELS["mistral"]["prompt"].format("a {b} c") == "<s>[INST] a {b} c [/INST]"


@pytest.mark.parametrize("key", sorted(REPLICATION_MODELS))
def test_replication_prompt_is_the_chat_template(key):
    # Needs the pinned tokenizer in the local Hugging Face cache.
    transformers = pytest.importorskip("transformers")
    cfg = REPLICATION_MODELS[key]
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            cfg["hf_model"], revision=cfg["revision"], local_files_only=True)
    except OSError:
        pytest.skip(f"{cfg['hf_model']} tokenizer not cached")
    q = "Контекст: {} Вопрос?"
    rendered = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True)
    assert cfg["prompt"].format(q) == rendered
    assert tok("x")["input_ids"][0] != tok.bos_token_id  # no BOS added on top


def test_replication_decoding_and_split():
    pipe = FakePipe("(B): Женщина")
    cfg = REPLICATION_MODELS["qwen"]
    assert ask(pipe, cfg, ["Q"], bs=16) == ["(B): Женщина"]
    kw = pipe.calls[0]
    assert kw["max_new_tokens"] == 512 and kw["do_sample"] is False and kw["num_beams"] == 1
    assert kw["repetition_penalty"] == 1.0
    assert all(c["padding_side"] == "left" for c in REPLICATION_MODELS.values())


def test_paper_models_get_no_extra_generation_kwargs():
    pipe = FakePipe("(a)")
    ask(pipe, PAPER_MODELS["mistral"], ["Q"], bs=16)
    assert "repetition_penalty" not in pipe.calls[0]


def test_paper_settings_stamp_unchanged():
    # Work folders stamped before the replication models must still resume.
    for key, cfg in PAPER_MODELS.items():
        assert list(settings_for(key, cfg, cfg["bs"])) == [
            "model", "hf_model", "revision", "prompt", "max_new_tokens", "bs"]
    s = settings_for("tlite", REPLICATION_MODELS["tlite"], 16)
    assert s["padding_side"] == "left" and s["generate_kwargs"] == {"repetition_penalty": 1.0}
    assert s["transformers"] == "4.46.3"
