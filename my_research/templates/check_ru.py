"""Every mechanical check of the Russian data, in one command. Run after every
edit of a worksheet, the lexicon or ru_unknowns.csv; it writes nothing.

  python my_research/templates/check_ru.py
  python my_research/templates/check_ru.py --category Age --category SES

1. Generation: everything `generate_ru.py --check` reports as blocking
   (template syntax, lexicon gaps, people it can't place).
2. Generated items (main, control and masc), each problem with its template
   and the number of items it affects:
   - formatting: capitals, sentence ends, spacing, doubled punctuation,
     unbalanced «» and (), Latin letters (except LATIN_OK), leftover slots;
   - answer options: empty, lowercase, ending in a period, or two the same;
   - em and en dashes (DECISIONS.md, "Punctuation");
   - «о» before a vowel, «об» before a consonant or е/ё/ю/я;
   - masculine words for the pair («двое», «оба», «один из них», …) in items
     where both people are feminine (DECISIONS.md, "Agreement and gender",
     PAIR slots), unless the word counts a following noun that isn't a pair
     noun («в обоих рассказах»);
   - control set: a past-tense verb or short adjective right after a name
     (nominative) that doesn't agree with the name;
   - words pymorphy3 doesn't know, except KNOWN_STEMS.
3. Worksheets and lexicon: dashes in the template text; mixed scripts, stray
   spaces and dashes in the lexicon; every lexicon case form against
   pymorphy3 (each changed word must have a parse in that case with the same
   lemma as the `nom` word; gen/dat/ins/prep must change some word unless the
   value is a prepositional phrase such as «в положении»).

Exit status 1 if anything is reported.
"""
import argparse
import re
import sys
from collections import Counter, defaultdict

import pymorphy3

import generate_ru as g

LATIN_OK = {"IT"}
# Right words pymorphy3 doesn't know, as lowercase prefixes (ё written е)
KNOWN_STEMS = ("трансгендерн", "цисгендерн", "бисексуал", "гетеросексуал", "парковщик",
               "брауни", "аспергер", "кардио")
TEXT = ("context", "question")
ANSWERS = ("ans0", "ans1", "ans2")
RU_COLUMNS = (*g.CASES, "answer", "with", "act", "who", "dressed")
PYMORPHY_CASE = {"gen": {"gent", "gen2"}, "dat": {"datv"}, "acc": {"accs"},
                 "ins": {"ablt"}, "prep": {"loct", "loc2"}}

WORD = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:-[А-Яа-яЁёA-Za-z]+)*")
DASH = re.compile(r"[—–]")
O_VOWEL = re.compile(r"(?<!\w)[Оо] (?=[аиоуэыАИОУЭЫ])")
OB_CONSONANT = re.compile(r"(?<!\w)[Оо]б (?=[бвгджзйклмнпрстфхцчшщеёюяБВГДЖЗЙКЛМНПРСТФХЦЧШЩЕЁЮЯ])")
# Masculine words for the pair that have a feminine form (обе, одна из них, …)
PAIR_MASC = re.compile(
    r"(?<!\w)(?:двое|двоих|двоим|двоими|оба|обоих|обоим|обоими"
    r"|(?:ни )?(?:один|одного|одному|одним|одном) из них"
    r"|(?:каждый|каждого|каждому|каждым|каждом) из них)(?!\w)", re.IGNORECASE)
# Nouns for the pair that have a feminine form. «Двое»/«оба» before any other
# noun counts that noun («обоих соперников»), not the pair.
PAIR_NOUNS = {"друг", "сосед", "приятель", "одноклассник", "однокурсник", "студент",
              "ученик", "знакомый", "товарищ"}
AGREEING = ("VERB", "ADJS", "PRTS")
FORMAT = [
    (re.compile(r"\s{2,}"), "double space"),
    (re.compile(r"\s[,.;:!?)»…]"), "space before punctuation"),
    (re.compile(r"[(«]\s"), "space after an opening bracket"),
    (re.compile(r"[,;:](?=[^\s\d])"), "no space after , ; or :"),
    (re.compile(r"[,;:]\s*[,.;:!?]|(?<!\.)\.\.(?!\.)|[!?]\s*[,;:.]"), "doubled punctuation"),
    (re.compile(r"[.!?]\s+[а-яё]"), "sentence starts lowercase"),
    (re.compile(r"\{\{|\}\}"), "leftover slot braces"),
    (re.compile(re.escape(g.MISSING)), "missing value"),
]

morph = pymorphy3.MorphAnalyzer()


class Report:
    def __init__(self):
        self.sections = defaultdict(lambda: defaultdict(Counter))  # section -> msg -> where -> n
        self.examples = {}

    def add(self, section, msg, where, example=None):
        self.sections[section][msg][where] += 1
        if example is not None:
            self.examples.setdefault((section, msg, where), example)

    def count(self):
        return sum(len(by_where) for msgs in self.sections.values() for by_where in msgs.values())

    def print(self):
        for section, msgs in self.sections.items():
            print(f"\n== {section} ({sum(len(w) for w in msgs.values())})")
            for msg, by_where in sorted(msgs.items()):
                for where, n in sorted(by_where.items()):
                    ex = self.examples.get((section, msg, where))
                    print(f"  {where}: {msg}" + (f"  [{n} items]" if n > 1 else "")
                          + (f"\n      {ex}" if ex else ""))


def snippet(text, i, width=35):
    return "…" + text[max(0, i - width):i + width] + "…"


def known(word):
    w = word.lower().replace("ё", "е")
    if morph.word_is_known(w):
        return True
    return all(morph.word_is_known(part) or part.startswith(KNOWN_STEMS)
               or not re.search(r"[а-я]", part) for part in w.split("-"))


def check_text(field, text, where, report):
    for pattern, msg in FORMAT:
        m = pattern.search(text)
        if m:
            report.add("formatting", f"{field}: {msg}", where, snippet(text, m.start()))
    if text[:1].islower():
        report.add("formatting", f"{field}: starts lowercase", where, text[:60])
    if text[-1:] not in ("?" if field == "question" else ".!?…»"):
        report.add("formatting", f"{field}: ends in {text[-1:]!r}", where, text[-60:])
    for a, b in ("«»", "()"):
        if text.count(a) != text.count(b):
            report.add("formatting", f"{field}: unbalanced {a}{b}", where, text[:80])
    for m in DASH.finditer(text):
        report.add("dashes", f"{field}: dash", where, snippet(text, m.start()))
    for pattern, msg in ((O_VOWEL, "«о» before a vowel"), (OB_CONSONANT, "«об» before a consonant")):
        for m in pattern.finditer(text):
            report.add("о/об", f"{field}: {msg}", where, snippet(text, m.start()))
    for w in WORD.findall(text):
        if any(re.search(r"[A-Za-z]", part) and part not in LATIN_OK for part in w.split("-")):
            report.add("formatting", f"{field}: Latin letters «{w}»", where)


def check_answers(r, where, report):
    seen = {}
    for k in ANSWERS:
        a = r[k]
        if not a.strip():
            report.add("answer options", f"{k} empty", where)
            continue
        if a[:1].islower():
            report.add("answer options", f"{k} starts lowercase", where, a)
        if a.endswith("."):
            report.add("answer options", f"{k} ends in a period", where, a)
        if DASH.search(a):
            report.add("dashes", f"{k}: dash", where, a)
        key = a.strip().lower().replace("ё", "е")
        if key in seen:
            report.add("answer options", f"{seen[key]} and {k} are the same", where, a)
        seen[key] = k


def genders(src, kind, lex):
    """Lexicon gender of the two people in an English sample."""
    out = []
    for k, (v, t) in src["answer_info"].items():
        if t == "unknown":
            continue
        if kind == "control":
            out.append(g.EN_NAME_GENDER.get(v))
        else:
            e = lex.entry(src["category"], int(src["question_index"]), v, src[k], g.Problems())
            out.append(e and e["gender"])
    return out


def check_pair_words(r, where, report):
    """Both people feminine: no masculine word for the pair."""
    for field in TEXT:
        for m in PAIR_MASC.finditer(r[field]):
            nxt = WORD.match(r[field], m.end() + 1)
            if nxt and not m.group(0).lower().endswith("из них"):
                p = morph.parse(nxt.group(0).lower())[0]
                if p.tag.POS == "NOUN" and p.normal_form not in PAIR_NOUNS:
                    continue
            report.add("masculine pair words, both people feminine", f"{field}: «{m.group(0)}»",
                       where, snippet(r[field], m.start()))


def check_name_agreement(r, names, gender, where, report):
    """Control set: a gendered word right after a name agrees with it.

    Only nominative names of the pair's gender count: «Александра» in a male
    pair is the genitive of «Александр».
    """
    want = {"m": "masc", "f": "femn"}[gender]
    for field in TEXT:
        tokens = re.findall(r"[А-Яа-яЁё]+|[^\sА-Яа-яЁё]", r[field])
        for i, tok in enumerate(tokens):
            if names.get(tok) != gender:
                continue
            for nxt in tokens[i + 1:i + 4]:
                if not re.match(r"[А-Яа-яЁё]", nxt):
                    break
                p = morph.parse(nxt.lower())[0]
                if p.tag.POS in AGREEING and "sing" in p.tag and p.tag.gender in ("masc", "femn"):
                    if p.tag.gender != want:
                        report.add("control agreement", f"{field}: «{tok} … {nxt}»", where,
                                   snippet(r[field], r[field].find(nxt)))
                    break


def check_generated(args, templates, lex, unknowns, problems, report):
    names = {r["nom"].strip(): r["gender"] for r in lex.rows if r["category"] == "*"}
    unknown_words, first_seen = Counter(), {}
    total = 0
    for cat in args.category or g.CATS:
        for kind in ["main", "control"] + (["masc"] if cat == "Gender_identity" else []):
            built, _, _ = g.build(cat, kind, templates, lex, unknowns, g.ROOT / "data", problems)
            total += len(built)
            for src, row, r in built:
                variant = row["variant"].strip()
                where = (f"{cat} q{src['question_index']}" + (f" ({variant})" if variant else "")
                         + ("" if kind == "main" else f" [{kind}]"))
                for field in TEXT:
                    check_text(field, r[field], where, report)
                    for w in WORD.findall(r[field]):
                        if not known(w):
                            unknown_words[w] += 1
                            first_seen.setdefault(w, where)
                check_answers(r, where, report)
                gs = genders(src, kind, lex)
                if all(x == "f" for x in gs):
                    check_pair_words(r, where, report)
                if kind == "control":
                    check_name_agreement(r, names, gs[0], where, report)
    for w, n in unknown_words.most_common():
        report.add("words pymorphy3 doesn't know (add to KNOWN_STEMS if right)",
                   f"«{w}» ({n}×)", first_seen[w])
    return total


def check_sources(templates, lex, unknowns, report):
    for cat, by_q in templates.items():
        for q, variants in by_q.items():
            for variant, row in variants.items():
                where = f"{cat} q{q}" + (f" ({variant})" if variant else "") + " [worksheet]"
                for f in g.FIELDS:
                    for m in DASH.finditer(row[f]):
                        report.add("dashes", f"{f}: dash", where, snippet(row[f], m.start()))
    for en, ru in unknowns.items():
        if DASH.search(ru):
            report.add("lexicon", f"ru_unknowns.csv: dash «{ru}»", en)
    for r in lex.rows:
        where = g.row_name(r)
        for c in RU_COLUMNS:
            v = r.get(c) or ""
            if v != v.strip() or "  " in v:
                report.add("lexicon", f"{c}: stray space «{v}»", where)
            if DASH.search(v):
                report.add("lexicon", f"{c}: dash «{v}»", where)
            for w in WORD.findall(v):
                if re.search(r"[A-Za-z]", w) and re.search(r"[А-Яа-яЁё]", w):
                    report.add("lexicon", f"{c}: mixed scripts «{w}»", where)
        check_cases(r, where, report)


def same_lemma(a, b):
    lemmas = {p.normal_form for p in morph.parse(a.lower())}
    return bool(lemmas & {p.normal_form for p in morph.parse(b.lower())})


def check_cases(r, where, report):
    nom = r["nom"].split()
    if not nom:
        return
    prep_phrase = morph.parse(nom[0].lower())[0].tag.POS == "PREP"
    for c, cases in PYMORPHY_CASE.items():
        form = (r.get(c) or "").split()
        if not form:
            continue
        if len(form) != len(nom):
            report.add("lexicon", f"{c}: «{r[c]}» has {len(form)} words, nom «{r['nom']}» {len(nom)}",
                       where)
            continue
        changed = [(n, f) for n, f in zip(nom, form) if n.lower() != f.lower()]
        if not changed and c != "acc" and not prep_phrase:
            report.add("lexicon", f"{c}: «{r[c]}» same as nom", where)
        for n, f in changed:
            for n_part, f_part in zip(n.split("-"), f.split("-")):
                if n_part.lower() == f_part.lower():
                    continue
                if not any(p.tag.case in cases and same_lemma(f_part, n_part)
                           for p in morph.parse(f_part.lower())):
                    report.add("lexicon", f"{c}: «{f_part}» isn't a {c} form of «{n_part}»", where)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--category", action="append", choices=g.CATS,
                    help="repeatable; default: all six")
    args = ap.parse_args()
    problems = g.Problems()
    templates, source = g.load_templates(problems)
    lex = g.Lexicon(g.HERE / "ru_lexicon.csv")
    unknowns = {r["en"]: r["ru"].strip() for r in g.read_csv(g.HERE / "ru_unknowns.csv")}
    report = Report()
    total = check_generated(args, templates, lex, unknowns, problems, report)
    check_sources(templates, lex, unknowns, report)

    print(f"Templates: {source}; {total} items checked")
    problems.report(g.Problems.BLOCKING)
    problems.report_gaps()
    report.print()
    n = problems.blocking() + report.count()
    print(f"\n{n} problem(s)." if n else "\nNo problems.")
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
