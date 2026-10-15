"""Normalise a Russian model response before upstream answer detection.

normalize_response (applied to the response only; the stored answer is
unchanged):
1. ё → е, in the response and (normalize_option) in the answer options:
   models often write «Артем» for «Артём», and some write ё where the
   option has е.
2. A spaced hyphen or en dash → «—», so one phrase form covers «Ответ - (A)»,
   «Ответ – (A)» and «Ответ — (A)». Hyphens inside words stay.
3. Option-shaped Cyrillic А/С → Latin A/C (normalize_letters). Cyrillic А and
   С look like Latin A and C, and upstream only knows Latin letters. Cyrillic
   В is left alone: it looks like Latin B but is the third Cyrillic letter, so
   a lone «В» is ambiguous and stays undetected.

"Option-shaped" means the forms upstream's a_list/c_list contain, such as
«(А)», «А)», «А:», «А.», with the letter capital or in brackets or quotes. A
bare «А»/«С», or one with a comma, is also a word («А другой…», «С учётом…»,
«а, значит,…»), so it maps only where it can't be one: at the end of a line,
after a colon or a dash, or after «вариант»/«ответ»/«буква».
"""
import re

CYR_TO_LAT = str.maketrans("АаСс", "AaCc")
# Upstream's a_list/c_list shapes, minus the bare letters, the comma forms and
# the Turkish "'dir" forms (handled separately or not applicable)
SHAPES = {
    f.format(x) for x in "AaCc"
    for f in ("{}.", "{})", "({})", "{}:", "({}):", "'{}'", "'{}'.", '"{}:', '"{}"', "'{}.'",
              "[{}]", '"{}."', "({}),", "({}).", '"{}".')
}
OPTION_WORDS = {"вариант", "варианта", "варианту", "вариантом", "варианте", "ответ",
                "ответа", "буква", "букву", "буквой", "пункт"}
BARE = re.compile(r"^[АС],?$")


def normalize_letters(text):
    parts = re.split(r"(\s+)", text)  # words at even indices, whitespace at odd
    for i in range(0, len(parts), 2):
        tok = parts[i]
        if not tok or len(tok) > 5:
            continue
        lat = tok.translate(CYR_TO_LAT)
        if lat == tok:
            continue
        if lat in SHAPES and (lat.strip("()[]'\".:,")[:1].isupper() or re.search(r"[()\[\]'\"]", tok)):
            parts[i] = lat
        elif BARE.match(tok) and bare_is_letter(parts, i):
            parts[i] = lat
    return "".join(parts)


def bare_is_letter(parts, i):
    """A bare «А»/«С» (or «А,») stands for an option, not a word."""
    space_after = parts[i + 1] if i + 1 < len(parts) else ""
    end_of_line = "\n" in space_after or not (parts[i + 2] if i + 2 < len(parts) else "")
    prev = parts[i - 2] if i >= 2 else ""
    return (end_of_line and not parts[i].endswith(",")) or prev.endswith(":") \
        or prev in ("—", "-", "–") or prev.lower() in OPTION_WORDS


def normalize_option(text):
    return text.replace("ё", "е").replace("Ё", "Е")


def normalize_response(text):
    text = normalize_option(text)
    text = re.sub(r"(?<=\S) [-–] (?=\S)", " — ", text)
    return normalize_letters(text)
