"""«с» -> «со» before a filled slot (generate_ru.SO)."""
from generate_ru import Problems, render


def entry(ins, gender="m"):
    return {"ins": ins, "gender": gender, "nom": ins}


def fill(text, a, b="Анной"):
    return render(text, {"NAME1": entry(a), "NAME2": entry(b, "f")}, Problems(), "test")


def test_before_s_consonant():
    assert fill("интервью с {{NAME1:ins}}", "стариком") == "интервью со стариком"


def test_before_sh_and_z_and_zh():
    assert fill("с {{NAME1:ins}}", "школьником") == "со школьником"
    assert fill("с {{NAME1:ins}}", "зрителем") == "со зрителем"
    assert fill("с {{NAME1:ins}}", "жрецом") == "со жрецом"


def test_before_lev():
    assert fill("с {{NAME1:ins}}", "Львом") == "со Львом"
    assert fill("С {{NAME1:ins}} мы", "Львом") == "Со Львом мы"
    assert fill("с {{NAME1:ins}}", "Лилией") == "с Лилией"


def test_capitalised_at_sentence_start():
    assert fill("{{NAME1:ins}} всё ясно.", "x") == "X всё ясно."
    assert fill("С {{NAME1:ins}} у меня", "строителем") == "Со строителем у меня"


def test_not_before_vowel_or_other_letters():
    assert fill("с {{NAME1:ins}}", "сыном") == "с сыном"
    assert fill("с {{NAME1:ins}}", "Сергеем") == "с Сергеем"
    assert fill("с {{NAME1:ins}}", "Анной") == "с Анной"


def test_not_inside_words():
    assert fill("вас {{NAME1:ins}}", "стариком") == "вас стариком"


def test_second_slot_too():
    assert fill("с {{NAME1:ins}} и с {{NAME2:ins}}", "Иваном", "стройной студенткой") == \
        "с Иваном и со стройной студенткой"
