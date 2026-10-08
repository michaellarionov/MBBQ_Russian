from ask_model_paper import PAPER_MODELS, ask, split_answer


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
