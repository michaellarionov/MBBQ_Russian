from collections import Counter

from generate_ru import NamePicker, Problems, assign, mention_order, render

MAN = {"category": "X", "q_id": "*", "en_value": "man", "gender": "m",
       "nom": "мужчина", "acc": "мужчину"}
WOMAN = {"category": "X", "q_id": "*", "en_value": "woman", "gender": "f",
         "nom": "женщина", "acc": "женщину"}
ROW = {"ambig_ctx": "{{NAME1:nom}} и {{NAME2:nom}} опоздали.", "neg_answer_slot": "NAME1"}


def person(pos, value):
    return {"pos": pos, "key": f"ans{pos}", "value": value}


def sample(cond, pol, label, context):
    return {"context_condition": cond, "question_polarity": pol, "label": label,
            "context": context}


def test_mention_order_ignores_value_inside_longer_value():
    ps = [person(0, "woman"), person(1, "trans woman")]
    order = mention_order("A trans woman and a woman were late.", ps)
    assert [p["value"] for p in order] == ["trans woman", "woman"]


def test_mention_order_hyphen_matches_space():
    ps = [person(0, "15 year old"), person(1, "40 year old")]
    order = mention_order("a 40-year-old and a 15-year-old", ps)
    assert [p["value"] for p in order] == ["40 year old", "15 year old"]


def test_ambiguous_follows_english_mention_order():
    ps = [person(1, "woman"), person(2, "man")]
    roles = assign(sample("ambig", "neg", 0, "a man and a woman were late"), ROW, ps,
                   Problems(), "X q1")
    assert roles["NAME1"]["value"] == "man"


def test_disambiguated_gold_fills_the_asked_slot():
    ps = [person(1, "woman"), person(2, "man")]
    ctx = "a man and a woman were late."
    neg = assign(sample("disambig", "neg", 1, ctx), ROW, ps, Problems(), "X q1")
    assert neg["NAME1"]["value"] == "woman"  # neg_answer_slot
    nonneg = assign(sample("disambig", "nonneg", 1, ctx), ROW, ps, Problems(), "X q1")
    assert nonneg["NAME2"]["value"] == "woman"


def test_render_agreement_and_sentence_start():
    p = Problems()
    text = "Ждали {{NAME2:acc}}. {{NAME1:nom}} {{NAME1?опоздал|опоздала}}."
    assert render(text, {"NAME1": MAN, "NAME2": WOMAN}, p, "X q1") == \
        "Ждали женщину. Мужчина опоздал."
    assert not p.gaps


def test_render_reports_missing_form():
    p = Problems()
    render("{{NAME1:ins}}", {"NAME1": MAN, "NAME2": WOMAN}, p, "X q1")
    assert p.gaps["X q* 'man'"][0] == {"ins"}


def test_name_picker_balanced_and_never_repeats_in_a_pair():
    names = {"f": [f"f{i}" for i in range(10)]}
    picker, use = NamePicker(names, "seed"), Counter()
    for _ in range(500):
        a = picker.take("f", set())
        b = picker.take("f", {a})
        assert a != b
        use.update([a, b])
    assert max(use.values()) - min(use.values()) <= 2
