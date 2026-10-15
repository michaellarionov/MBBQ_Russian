"""Tests for the {{PAIR?masc|fem}} slot in generate_ru.py (run: python -m pytest)."""
import generate_ru as g

M = {"category": "x", "q_id": "*", "en_value": "m", "gender": "m", "nom": "Иван"}
F = {"category": "x", "q_id": "*", "en_value": "f", "gender": "f", "nom": "Анна"}
F2 = dict(F, en_value="f2", nom="Мария")
T = "{{PAIR?Двое друзей|Две подруги}} — {{NAME1:nom}} и {{NAME2:nom}} — пришли."


def run(n1, n2, text=T):
    p = g.Problems()
    return g.render(text, {"NAME1": n1, "NAME2": n2}, p, "test"), p


def test_two_women_feminine():
    assert run(F, F2)[0] == "Две подруги — Анна и Мария — пришли."


def test_two_men_masculine():
    assert run(M, dict(M, nom="Лев"))[0] == "Двое друзей — Иван и Лев — пришли."


def test_mixed_pair_masculine_either_order():
    assert run(M, F)[0].startswith("Двое друзей")
    assert run(F, M)[0].startswith("Двое друзей")


def test_capitalised_at_sentence_start_only():
    text = "Вчера пришли {{PAIR?двое друзей|две подруги}}. {{PAIR?двое|две}} остались."
    assert run(F, F2, text)[0] == "Вчера пришли две подруги. Две остались."


def test_missing_gender_is_a_gap():
    out, p = run(F, dict(F2, gender=""))
    assert g.MISSING in out and p.gaps


def test_pair_allowed_in_ambiguous_context():
    assert not g.AGREEMENT.search(T)


def test_pair_first_does_not_break_mention_order():
    row = {"ambig_ctx": T}
    sample = {"context": "Two friends, a woman and a man, came."}
    persons = [{"pos": 1, "value": "man"}, {"pos": 2, "value": "woman"}]
    roles = g.order_roles(sample, row, persons)
    assert set(roles) == {"NAME1", "NAME2"} and roles["NAME1"]["value"] == "woman"
