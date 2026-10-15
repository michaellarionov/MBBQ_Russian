"""Each check in check_ru.py fires on a broken example (run: python -m pytest)."""
import check_ru as c


def found(fn, *args):
    report = c.Report()
    fn(*args, "X q1", report)
    return {msg for msgs in report.sections.values() for msg in msgs}


def test_clean_text_passes():
    assert not found(c.check_text, "context",
                     "Двое друзей, Ева и Мария, говорили об этом в IT-компании.")


def test_formatting():
    msgs = found(c.check_text, "context", "мария пришла ,  а ⟨?⟩ ушла..")
    assert {"context: starts lowercase", "context: double space",
            "context: space before punctuation", "context: missing value",
            "context: doubled punctuation"} <= msgs
    assert "question: ends in '.'" in found(c.check_text, "question", "Кто пришёл.")
    assert "context: sentence starts lowercase" in found(c.check_text, "context", "Он пришёл. она ушла.")
    assert "context: Latin letters «Anna»" in found(c.check_text, "context", "Пришла Anna.")


def test_dashes_and_o_ob():
    msgs = found(c.check_text, "context", "Двое — Ева и Мария — говорили о истории и об сделке.")
    assert {"context: dash", "context: «о» before a vowel",
            "context: «об» before a consonant"} <= msgs
    assert not found(c.check_text, "context", "Говорили об истории и о ёлке.")


def test_answers():
    r = {"ans0": "Неизвестно", "ans1": "мужчина", "ans2": "Мужчина."}
    assert {"ans1 starts lowercase", "ans2 ends in a period"} <= found(c.check_answers, r)
    r = {"ans0": "Неизвестно", "ans1": "Артём", "ans2": "Артем"}
    assert "ans1 and ans2 are the same" in found(c.check_answers, r)


def test_pair_words():
    r = {"context": "Двое друзей пришли. Один из них ушёл.", "question": "Кто?"}
    assert {"context: «Двое»", "context: «Один из них»"} <= found(c.check_pair_words, r)
    r = {"context": "Пиво упоминалось в обоих рассказах.", "question": "Кто?"}
    assert not found(c.check_pair_words, r)


def test_name_agreement():
    names = {"Анна": "f", "Александра": "f", "Иван": "m", "Александр": "m"}
    r = {"context": "Анна быстро пришёл, а Иван ушёл.", "question": "Кто?"}
    assert found(c.check_name_agreement, r, names, "f") == {"context: «Анна … пришёл»"}
    # «Александра» is the genitive of «Александр» in a male pair
    r = {"context": "У Ивана и Александра была консультация.", "question": "Кто?"}
    assert not found(c.check_name_agreement, r, names, "m")


def test_unknown_words():
    assert c.known("Трансгендерная") and c.known("когда-нибудь") and c.known("тёмно-синем")
    assert not c.known("мужщина")


def test_lexicon_cases():
    row = {"nom": "пожилой мужчина", "gen": "пожилого мужчины", "dat": "пожилой мужчина",
           "ins": "пожилым мужчиной", "prep": "пожилом мужчиной", "acc": "пожилого мужчину"}
    msgs = found(c.check_cases, row)
    assert msgs == {"dat: «пожилой мужчина» same as nom",
                    "prep: «мужчиной» isn't a prep form of «мужчина»"}
    attr = {"nom": "в положении", "gen": "в положении", "dat": "в положении"}
    assert not found(c.check_cases, attr)
