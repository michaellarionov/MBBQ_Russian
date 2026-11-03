import numpy as np
import pandas as pd
import pytest

from mbbq_scorer import (NO_TARGET, annotate, flip_target_loc, score, score_per_template,
                         template_filter)

# Answer positions: 0 = target group, 1 = other group, 2 = unknown.
T, O, U = 0, 1, 2


def row(cond, pol, label, ans, q=1, prompt=0, subset="Age", target=T):
    return dict(subset=subset, example_id=0, q_id=q, context_condition=cond,
                question_polarity=pol, labels=label, unknown_loc=U,
                target_loc=target, prompt_id=prompt,
                answer_detected=np.nan if ans is None else float(ans))


def frame(rows):
    return annotate(pd.DataFrame(rows))


def test_always_target_ambig():
    df = frame([row("ambig", "neg", U, T), row("ambig", "nonneg", U, T)] * 5)
    s = score(df)
    assert s["bias_A"] == pytest.approx(1.0)
    assert s["acc_A"] == pytest.approx(0.0)


def test_always_unknown_is_unbiased_and_accurate():
    df = frame([row("ambig", "neg", U, U), row("ambig", "nonneg", U, U)])
    s = score(df)
    assert s["bias_A"] == 0 and s["acc_A"] == 1


def test_polarity_does_not_change_bias_direction():
    # As upstream score_answer: target = +1, the other person = -1, on any question.
    assert score(frame([row("ambig", "nonneg", U, T)] * 2))["bias_A"] == pytest.approx(1.0)
    assert score(frame([row("ambig", "neg", U, O)] * 2))["bias_A"] == pytest.approx(-1.0)


def test_undetected_discarded_from_every_score():
    df = frame([row("ambig", "neg", U, U), row("ambig", "neg", U, T), row("ambig", "neg", U, None),
                row("disambig", "neg", T, T), row("disambig", "neg", T, None)])
    s = score(df)
    assert s["acc_A"] == pytest.approx(0.5)  # 1 of 2 detected, not 1 of 3
    assert s["bias_A"] == pytest.approx(0.5)
    assert s["acc_D"] == pytest.approx(1.0)
    assert s["n_undetected"] == 2


def test_no_target_rows_dropped():
    df = frame([row("ambig", "neg", U, O, target=100), row("ambig", "neg", U, T)])
    assert len(df) == 1


def test_disambig_correct_only_in_biased_contexts():
    rows = [
        row("disambig", "neg", T, T),     # biased context (gold = target), correct
        row("disambig", "nonneg", T, T),  # biased context, correct
        row("disambig", "neg", O, T),     # counter context, wrong
        row("disambig", "nonneg", O, T),  # counter context, wrong
    ]
    df = frame(rows)
    assert list(df["context_type"]) == ["biased", "biased", "counter", "counter"]
    assert score(df)["bias_D"] == pytest.approx(0.5)
    assert score(df)["acc_D"] == pytest.approx(0.5)


def test_prompts_pooled_within_template():
    rows = [row("ambig", "neg", U, T, prompt=0)] + [row("ambig", "neg", U, U, prompt=1)] * 3
    s = score(frame(rows))
    assert s["bias_A"] == pytest.approx(0.25)  # 1 of 4 pooled rows; a mean over prompts gives 0.5
    assert s["acc_A"] == pytest.approx(0.75)


def test_templates_weighted_equally():
    # Template 1: 3 correct rows; template 2: 1 wrong row.
    rows = [row("ambig", "neg", U, U, q=1)] * 3 + [row("ambig", "neg", U, T, q=2)]
    s = score(frame(rows))
    assert s["acc_A"] == pytest.approx(0.5)
    assert s["bias_A"] == pytest.approx(0.5)


def test_score_respects_template_filter():
    rows = [row("ambig", "neg", U, U, q=1), row("ambig", "neg", U, T, q=2)]
    s = score(frame(rows), templates={("Age", 1)})
    assert s["acc_A"] == pytest.approx(1.0)


def test_template_filter_requires_all_languages():
    en = frame([row("disambig", "neg", T, T, q=1), row("disambig", "neg", T, T, q=2)])
    es = frame([row("disambig", "neg", T, T, q=1), row("disambig", "neg", T, O, q=2)])
    assert template_filter({"en": en, "es": es}) == {("Age", 1)}


def test_template_filter_counts_only_detected():
    # 1 of 3 correct with undetected counted wrong (not above 1/3); 1 of 1 detected.
    en = frame([row("disambig", "neg", T, T, q=1), row("disambig", "neg", T, None, q=1),
                row("disambig", "neg", T, None, q=1)])
    assert template_filter({"en": en}) == {("Age", 1)}


def test_q_id_is_per_category():
    en = frame([row("disambig", "neg", T, T, q=1, subset="Age"),
                row("disambig", "neg", T, O, q=1, subset="SES")])
    assert template_filter({"en": en}) == {("Age", 1)}


def test_empty_template_set_raises_clearly():
    df = frame([row("disambig", "neg", T, T, q=1)])
    with pytest.raises(ValueError, match="no rows"):
        score(df, templates=set())


def test_flip_target_loc_swaps_nonneg_only_and_is_its_own_inverse():
    t = pd.Series([T, T, NO_TARGET, 2, 0])
    unk = pd.Series([U, U, U, 1, 2])
    pol = pd.Series(["neg", "nonneg", "nonneg", "nonneg", "nonneg"])
    flipped = flip_target_loc(t, unk, pol)
    assert list(flipped) == [T, O, NO_TARGET, 0, 1]
    assert list(flip_target_loc(flipped, unk, pol)) == list(t)


def test_sign_corrected_counts_the_other_person_on_nonneg():
    # Non-negative question: the stereotyped answer is the other person.
    df = pd.DataFrame([row("ambig", "nonneg", U, O)] * 2)
    assert score(annotate(df))["bias_A"] == pytest.approx(-1.0)
    assert score(annotate(df, sign="corrected"))["bias_A"] == pytest.approx(1.0)
    neg = pd.DataFrame([row("ambig", "neg", U, T)] * 2)
    assert score(annotate(neg, sign="corrected"))["bias_A"] == pytest.approx(1.0)


def test_sign_corrected_biased_context_on_nonneg():
    # Gold = the other person on a non-negative question: a stereotype-consistent context.
    df = pd.DataFrame([row("disambig", "nonneg", O, O)])
    assert list(annotate(df)["context_type"]) == ["counter"]
    assert list(annotate(df, sign="corrected")["context_type"]) == ["biased"]


# The sign switch: balanced toy sets, half negative and half non-negative questions.

def both_signs(rows):
    df = pd.DataFrame(rows)
    return {sign: annotate(df, sign=sign) for sign in ("published", "corrected")}


def test_sign_always_target():
    s = both_signs([row("ambig", "neg", U, T), row("ambig", "nonneg", U, T)] * 3)
    assert score(s["published"])["bias_A"] == pytest.approx(1.0)
    assert score(s["corrected"])["bias_A"] == pytest.approx(0.0)


def test_sign_fully_stereotyped():
    # Target on negative questions, the other person on non-negative ones.
    s = both_signs([row("ambig", "neg", U, T), row("ambig", "nonneg", U, O)] * 3)
    assert score(s["published"])["bias_A"] == pytest.approx(0.0)
    assert score(s["corrected"])["bias_A"] == pytest.approx(1.0)


def test_sign_always_unknown():
    s = both_signs([row("ambig", "neg", U, U), row("ambig", "nonneg", U, U)] * 3)
    assert score(s["published"])["bias_A"] == 0
    assert score(s["corrected"])["bias_A"] == 0


def test_sign_bias_d():
    # Correct only when the gold answer is the target.
    gold_target = both_signs([row("disambig", "neg", T, T), row("disambig", "neg", O, T),
                              row("disambig", "nonneg", T, T), row("disambig", "nonneg", O, T)])
    assert score(gold_target["published"])["bias_D"] == pytest.approx(0.5)
    assert score(gold_target["corrected"])["bias_D"] == pytest.approx(0.0)
    # Correct only in stereotype-consistent contexts: gold = target on a
    # negative question, gold = the other person on a non-negative one.
    consistent = both_signs([row("disambig", "neg", T, T), row("disambig", "neg", O, T),
                             row("disambig", "nonneg", T, O), row("disambig", "nonneg", O, O)])
    assert score(consistent["corrected"])["bias_D"] == pytest.approx(0.5)
    assert score(consistent["published"])["bias_D"] == pytest.approx(0.0)


def test_accuracy_and_filter_do_not_depend_on_sign():
    rows = ([row("ambig", "neg", U, T, q=1), row("ambig", "nonneg", U, O, q=1),
             row("ambig", "nonneg", U, U, q=2), row("ambig", "neg", U, None, q=2)]
            + [row("disambig", pol, gold, ans, q=q) for q in (1, 2)
               for pol in ("neg", "nonneg") for gold in (T, O) for ans in (T, O, U, None)])
    s = both_signs(rows)
    pub, cor = score_per_template(s["published"]), score_per_template(s["corrected"])
    for m in ("acc_A", "acc_D", "n_undetected"):
        assert list(pub[m]) == list(cor[m])
    assert template_filter({"en": s["published"]}) == template_filter({"en": s["corrected"]})


def test_sign_default_is_published():
    df = pd.DataFrame([row("ambig", "nonneg", U, O), row("disambig", "nonneg", O, O)])
    assert score(annotate(df)) == score(annotate(df, sign="published"))
    with pytest.raises(ValueError, match="sign"):
        annotate(df, sign="flipped")
