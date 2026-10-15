"""Generate the Russian MBBQ data by replaying every English sample.

For each sample in data/<Category>[_control]_en.jsonl, take the Russian
template for its question_index, fill the slots from ru_lexicon.csv and keep
the English sample's answer order, label, tags, polarity and example_id. The
Russian data stays parallel to English item by item, and mbbq.py and the
scorer work unchanged (tags stay English).

Templates: ru_templates.csv if it exists (the rows kept after the vote),
otherwise the six worksheet_<Cat>_ru.csv files, minus rows with vote_keep=no.
Template row per sample (DECISIONS.md, "Slots and lexicon values"):
- control set: the `control` row if the template has one;
- BBQ version b: the `b` row if the template has one;
- otherwise the main row. `masc` rows go to Gender_identity_ru_masc.jsonl only,
  which holds just the templates that have one (DECISIONS.md, "Controlled
  gender-leak comparison").

People to slots: in a disambiguated sample the gold answer fills the slot the
question asks about (`neg_answer_slot` for a negative question, the other slot
for a non-negative one). In an ambiguous sample the person the English context
mentions first fills the first slot of the Russian ambiguous context.

Control names (DECISIONS.md, "Control-set names"): each group of samples that
shares an English name pair gets two different Russian names of the pair's
gender. Names are dealt from a seeded shuffle of the 10 per gender, so every
name is used about equally often and regenerating gives the same names.

Usage (from the fork root):
  python my_research/templates/generate_ru.py --check   # list every problem, write nothing
  python my_research/templates/generate_ru.py           # write data/<Cat>[_control]_ru.jsonl
  python my_research/templates/generate_ru.py --category Age --category SES
A file is written only if all its samples generate without problems.
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CATS = ["Age", "Disability_status", "Gender_identity", "Physical_appearance",
        "SES", "Sexual_orientation"]
FIELDS = ("ambig_ctx", "disambig_ctx", "q_neg", "q_nonneg")
CASES = ("nom", "gen", "dat", "acc", "ins", "prep")
# {{NAME1:gen}} -> a form column of the lexicon row; {{NAME1?ушёл|ушла}} -> by gender
# {{PAIR?двое друзей|две подруги}} -> feminine only if both people are `f`
SLOT = re.compile(r"\{\{(?:(NAME[12])(?::(\w+)|\?([^|{}]*)\|([^{}]*))"
                  r"|PAIR\?([^|{}]*)\|([^{}]*))\}\}")
AGREEMENT = re.compile(r"\{\{NAME[12]\?")
SENTENCE_START = re.compile(r"(^|[.!?…]\s+)$")
MISSING = "⟨?⟩"
# «с» -> «со» before с/з/ш/ж + consonant («со стариком», «со школьником») and
# before «льв»/«льд» («со Львом»), applied after the slots are filled, since
# only the value shows which it needs
SO = re.compile(r"(?<!\w)([Сс])(?= (?:[сзшжСЗШЖ][бвгджзйклмнпрстфхцчшщ]|[Лл]ь[вд]))")
# The 60 English control names (US 2022 top 30 per gender); every English
# control pair is same-gender.
EN_NAME_GENDER = {
    **dict.fromkeys(
        "Abigail Amelia Aria Ava Avery Camila Charlotte Chloe Eleanor Elizabeth "
        "Ella Emily Emma Evelyn Gianna Harper Hazel Isabella Layla Lily Luna Mia "
        "Mila Nora Olivia Penelope Scarlett Sofia Sophia Violet".split(), "f"),
    **dict.fromkeys(
        "Aiden Alexander Asher Benjamin Daniel Elijah Ethan Ezra Henry Hudson Jack "
        "Jackson James John Joseph Leo Levi Liam Luca Lucas Mason Mateo Michael "
        "Noah Oliver Owen Samuel Sebastian Theodore William".split(), "m"),
}


def other(slot):
    return "NAME2" if slot == "NAME1" else "NAME1"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Problems:
    """Problems by kind, each message with the number of samples it affects."""

    BLOCKING = ("template", "missing lexicon row", "ambiguous lexicon rows",
                "missing unknown phrasing", "mention order")

    def __init__(self):
        self.by_kind = defaultdict(Counter)
        self.gaps = defaultdict(lambda: (set(), set()))  # lexicon row -> (columns, templates)
        self.current = 0  # blocking problems in the sample being built

    def add(self, kind, msg):
        self.by_kind[kind][msg] += 1
        if kind in self.BLOCKING:
            self.current += 1

    def gap(self, e, column, where):
        """A lexicon row lacks a column that a template needs."""
        cols, used_by = self.gaps[row_name(e)]
        cols.add(column)
        used_by.add(where)
        self.current += 1

    def blocking(self):
        return sum(len(self.by_kind[k]) for k in self.BLOCKING) + len(self.gaps)

    def report_gaps(self):
        if not self.gaps:
            return
        print(f"\nlexicon gaps that block generation ({len(self.gaps)} rows):")
        order = (*CASES, "gender", "answer")
        for name, (cols, used_by) in sorted(self.gaps.items()):
            cols = sorted(cols, key=lambda c: (order.index(c) if c in order else len(order), c))
            print(f"  {name}: no {', '.join(cols)}  (needed by {', '.join(sorted(used_by))})")

    def report(self, kinds):
        for kind in kinds:
            msgs = self.by_kind.get(kind)
            if not msgs:
                continue
            print(f"\n{kind} ({len(msgs)}):")
            for msg, n in sorted(msgs.items()):
                print(f"  {msg}" + (f"  [{n} samples]" if n > 1 else ""))


def load_templates(problems):
    """{category: {q_id: {variant: row}}} and the source used."""
    path = HERE / "ru_templates.csv"
    if path.exists():
        rows, source = read_csv(path), path.name
    else:
        rows, source = [], "worksheet_<Cat>_ru.csv"
        for cat in CATS:
            rows += read_csv(HERE / f"worksheet_{cat}_ru.csv")
    out = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r.get("vote_keep", "").strip().lower() == "no":
            continue
        cat, q, variant = r["category"], int(r["q_id"]), r["variant"].strip()
        where = f"{cat} q{q}" + (f" ({variant})" if variant else "")
        if variant in out[cat][q]:
            problems.add("template", f"{where}: duplicate row")
        if r["neg_answer_slot"] not in ("NAME1", "NAME2"):
            problems.add("template", f"{where}: neg_answer_slot is {r['neg_answer_slot']!r}")
        for field in FIELDS:
            text = r[field]
            if "{{" in SLOT.sub("", text) or "}}" in SLOT.sub("", text):
                problems.add("template", f"{where} {field}: malformed slot")
            # Agreement with one person before the disambiguating context would
            # reveal who did what (DECISIONS.md, "Agreement and gender").
            if field != "disambig_ctx" and AGREEMENT.search(text):
                problems.add("template", f"{where} {field}: agreement slot not allowed here")
        out[cat][q][variant] = r
    for cat, by_q in out.items():
        for q, variants in by_q.items():
            if "" not in variants:
                problems.add("template", f"{cat} q{q}: variant rows but no main row")
    return out, source


class Lexicon:
    def __init__(self, path):
        self.rows = read_csv(path)
        self.index = defaultdict(list)
        self.names = defaultdict(list)  # gender -> Russian control names, in file order
        for r in self.rows:
            self.index[(r["category"], r["q_id"], r["en_value"])].append(r)
            if r["category"] == "*":
                self.names[r["gender"]].append(r["en_value"])
        self.used = set()

    def entry(self, cat, q, value, en_answer, problems):
        """The row for a value: (cat, q) first, then (cat, *), then (*, *).

        Several rows with the same key (a head noun that varies within the
        template) are told apart by `en_answers`, which must list the sample's
        English answer in exactly one of them.
        """
        rows = (self.index.get((cat, str(q), value)) or self.index.get((cat, "*", value))
                or self.index.get(("*", "*", value)))
        if not rows:
            problems.add("missing lexicon row", f"{cat} q{q} '{value}' (English answer: {en_answer})")
            return None
        if len(rows) > 1:
            rows = [r for r in rows
                    if en_answer in (a.strip() for a in r["en_answers"].split("|"))]
            if len(rows) != 1:
                problems.add("ambiguous lexicon rows",
                             f"{cat} q{q} '{value}': {len(rows)} rows list '{en_answer}' in en_answers")
                return None
        self.used.add(id(rows[0]))
        return rows[0]


def row_name(e):
    return f"{e['category']} q{e['q_id']} '{e['en_value']}'"


class NamePicker:
    """Deals Russian names per gender from seeded shuffles of the 10 names."""

    def __init__(self, names, seed):
        self.names, self.seed = names, seed
        self.queue, self.rounds = defaultdict(list), Counter()

    def take(self, gender, avoid):
        q = self.queue[gender]
        if len(q) < 2:
            deck = list(self.names[gender])
            random.Random(f"{self.seed}-{gender}-{self.rounds[gender]}").shuffle(deck)
            self.rounds[gender] += 1
            q.extend(deck)
        i = next(i for i, n in enumerate(q) if n not in avoid)
        return q.pop(i)


def value_pattern(value):
    """Whole-word, case-insensitive match; space and hyphen interchangeable."""
    body = re.sub(r"\\[ -]", "[ -]", re.escape(value))
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def mention_order(context, persons):
    """The two persons sorted by first mention in the English context.

    A match inside the other person's longer value doesn't count ("woman"
    inside "trans woman").
    """
    spans = {p["pos"]: [m.span() for m in value_pattern(p["value"]).finditer(context)]
             for p in persons}
    first = {}
    for p in persons:
        rival = spans[next(o["pos"] for o in persons if o is not p)]
        own = [s for s in spans[p["pos"]]
               if not any(a <= s[0] and s[1] <= b and (b - a) > (s[1] - s[0]) for a, b in rival)]
        if not own:
            return None
        first[p["pos"]] = own[0][0]
    return sorted(persons, key=lambda p: first[p["pos"]])


def assign(sample, row, persons, problems, where):
    """{slot: person}, or None if the people can't be placed."""
    if sample["context_condition"] == "disambig":
        neg = row["neg_answer_slot"]
        gold_slot = neg if sample["question_polarity"] == "neg" else other(neg)
        gold = [p for p in persons if p["pos"] == sample["label"]]
        if len(gold) != 1:
            problems.add("template", f"{where}: disambiguated sample whose label isn't a person")
            return None
        roles = {gold_slot: gold[0], other(gold_slot): next(p for p in persons if p is not gold[0])}
        by_mention = order_roles(sample, row, persons)
        if by_mention and by_mention != roles:
            problems.add("warning: order differs from English", where)
        return roles
    roles = order_roles(sample, row, persons)
    if roles is None:
        problems.add("mention order", f"{where}: can't find both people in the English context")
    return roles


def order_roles(sample, row, persons):
    order = mention_order(sample["context"], persons)
    if order is None:
        return None
    named = [m.group(1) for m in SLOT.finditer(row["ambig_ctx"]) if m.group(1)]  # skip PAIR
    first = named[0] if named else "NAME1"
    return {first: order[0], other(first): order[1]}


def render(text, slots, problems, where):
    def sub(m):
        if m.group(1) is None:  # PAIR: agrees with both people, so it cues neither
            pair = [slots["NAME1"], slots["NAME2"]]
            if any(e is None for e in pair):
                return MISSING
            for e in pair:
                if e["gender"] not in ("m", "f"):
                    problems.gap(e, "gender", where)
            if any(e["gender"] not in ("m", "f") for e in pair):
                return MISSING
            out = m.group(6) if all(e["gender"] == "f" for e in pair) else m.group(5)
            if SENTENCE_START.search(m.string[:m.start()]):
                out = out[:1].upper() + out[1:]
            return out
        e = slots[m.group(1)]
        if e is None:
            return MISSING
        if m.group(2):
            out = e.get(m.group(2), "").strip()
            if not out:
                problems.gap(e, m.group(2), where)
                return MISSING
        else:
            if e["gender"] not in ("m", "f"):
                problems.gap(e, "gender", where)
                return MISSING
            out = m.group(3) if e["gender"] == "m" else m.group(4)
        if SENTENCE_START.search(m.string[:m.start()]):
            out = out[:1].upper() + out[1:]
        return out
    return SO.sub(r"\1о", SLOT.sub(sub, text).strip())


def convert(sample, row, lex, unknowns, names, problems):
    """The Russian sample, or None if it has blocking problems."""
    cat, q = sample["category"], int(sample["question_index"])
    variant = row["variant"].strip()
    where = f"{cat} q{q}" + (f" ({variant})" if variant else "")
    problems.current = 0
    persons = [{"pos": int(k[-1]), "key": k, "value": v, "tag": t, "en_answer": sample[k]}
               for k, (v, t) in sample["answer_info"].items() if t != "unknown"]
    if len(persons) != 2:
        problems.add("template", f"{where}: sample with {len(persons)} people")
        return None
    roles = assign(sample, row, persons, problems, where)
    if roles is None:
        return None
    for p in persons:
        if names is not None:  # control set: the Russian name chosen for this English name
            p["entry"] = lex.entry("*", "*", names[p["value"]], p["en_answer"], problems)
        else:
            p["entry"] = lex.entry(cat, q, p["value"], p["en_answer"], problems)
    slots = {slot: p["entry"] for slot, p in roles.items()}
    context = render(row["ambig_ctx"], slots, problems, where)
    if sample["context_condition"] == "disambig":
        context += " " + render(row["disambig_ctx"], slots, problems, where)
    field = "q_neg" if sample["question_polarity"] == "neg" else "q_nonneg"
    question = render(row[field], slots, problems, where)
    out = dict(sample, context=context, question=question, answer_info={})
    for k, (v, t) in sample["answer_info"].items():
        if t == "unknown":
            ru = unknowns.get(v, "")
            short = ru
            if not ru:
                problems.add("missing unknown phrasing", f"'{v}' (ru_unknowns.csv)")
        else:
            e = next(p["entry"] for p in persons if p["key"] == k)
            ru = (e or {}).get("answer", "").strip()
            short = (e or {}).get("nom", "").strip() or ru
            if e and not ru:
                problems.gap(e, "answer", where)
        out[k] = ru
        out["answer_info"][k] = [short, t]  # tags stay English for mbbq.py and the scorer
    return None if problems.current else out


def control_names(samples, lex, seed, problems):
    """{example_id: {English name: Russian name}}, one pair per group of samples."""
    picker, out, prev, current = NamePicker(lex.names, seed), {}, None, None
    for s in samples:
        en = [v for v, t in s["answer_info"].values() if t != "unknown"]
        key = (s["question_index"], s["additional_metadata"].get("version"), tuple(sorted(en)))
        if key != prev:
            current, chosen = {}, set()
            for name in sorted(en):
                gender = EN_NAME_GENDER.get(name)
                if gender is None:
                    problems.add("template", f"control name {name!r} has no known gender")
                    gender = "f"
                current[name] = picker.take(gender, chosen)
                chosen.add(current[name])
            prev = key
        out[s["example_id"]] = current
    return out


def pick_row(variants, sample, control, masc):
    if masc:
        return variants.get("masc")
    if control and "control" in variants:
        return variants["control"]
    if sample["additional_metadata"].get("version") == "b" and "b" in variants:
        return variants["b"]
    return variants.get("")


def out_name(cat, kind):
    return f"{cat}{'_control' if kind == 'control' else ''}_ru{'_masc' if kind == 'masc' else ''}.jsonl"


def build(cat, kind, templates, lex, unknowns, data_dir, problems):
    """[(English sample, template row, Russian sample)], skipped, failed."""
    control, masc = kind == "control", kind == "masc"
    src = Path(data_dir) / f"{cat}{'_control' if control else ''}_en.jsonl"
    samples = [json.loads(line) for line in open(src, encoding="utf-8")]
    names = control_names(samples, lex, cat, problems) if control else None
    out, skipped, failed = [], 0, 0
    for s in samples:
        row = pick_row(templates.get(cat, {}).get(int(s["question_index"]), {}), s, control, masc)
        if row is None:  # not kept in the vote, or no masc row
            skipped += 1
            continue
        r = convert(s, row, lex, unknowns, names and names[s["example_id"]], problems)
        if r is None:
            failed += 1
        else:
            out.append((s, row, r))
    return out, skipped, failed


def generate(cat, kind, templates, lex, unknowns, args, problems):
    masc = kind == "masc"
    dst = Path(args.out) / out_name(cat, kind)
    built, skipped, failed = build(cat, kind, templates, lex, unknowns, args.data, problems)
    out = [r for _, _, r in built]
    status = f"{len(out)} ok, {failed} with problems, {skipped} skipped (template not kept"
    status += ", or no masc row)" if masc else ")"
    if failed or args.check:
        print(f"  {dst.name}: {status}{'' if args.check else '; not written'}")
        return
    with open(dst, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {dst.name}: wrote {len(out)}, skipped {skipped}")


def lexicon_report(lex, problems):
    """Rows that generate but still have empty columns, and rows nothing used."""
    incomplete, unused = [], []
    for r in lex.rows:
        if id(r) not in lex.used:
            unused.append(f"{row_name(r)} [{r['status']}]")
            continue
        if row_name(r) in problems.gaps:
            continue  # already listed as blocking
        gaps = [c for c in (*CASES, "gender", "answer") if not r.get(c, "").strip()]
        if gaps:
            incomplete.append(f"{row_name(r)}: no {', '.join(gaps)}")
    for title, items in (("other used rows with empty columns (not needed by the "
                          "current templates; LEXICON_GUIDE.md asks for them)", incomplete),
                         ("lexicon rows no sample used", unused)):
        if items:
            print(f"\n{title} ({len(items)}):")
            for i in items:
                print(f"  {i}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", action="append", choices=CATS,
                    help="repeatable; default: all six")
    ap.add_argument("--check", action="store_true", help="report problems, write nothing")
    ap.add_argument("--data", default=ROOT / "data", help="folder with the English jsonl")
    ap.add_argument("--out", default=ROOT / "data", help="folder for the Russian jsonl")
    args = ap.parse_args()
    problems = Problems()
    templates, source = load_templates(problems)
    lex = Lexicon(HERE / "ru_lexicon.csv")
    unknowns = {r["en"]: r["ru"].strip() for r in read_csv(HERE / "ru_unknowns.csv")}
    print(f"Templates: {source}")
    for cat in args.category or CATS:
        kinds = ["main", "control"] + (["masc"] if cat == "Gender_identity" else [])
        for kind in kinds:
            generate(cat, kind, templates, lex, unknowns, args, problems)
    problems.report(Problems.BLOCKING)
    problems.report_gaps()
    problems.report(["warning: order differs from English"])
    if args.check and not args.category:
        lexicon_report(lex, problems)
    n = problems.blocking()
    print(f"\n{n} distinct problem(s)." if n else "\nNo problems.")
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
