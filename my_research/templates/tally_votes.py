"""Tally the stereotype vote and fill `vote_keep` (DECISIONS.md, "Stereotype vote").

  python my_research/templates/tally_votes.py VOTES.csv --rename=-=V3 [--check]

VOTES.csv is the Google Forms export. Keep it out of git: it has one row per
voter with their background, and the voters consented to the use of their
answers in aggregate only. `--rename TYPED=CODE` fixes a voter code typed
wrong in the form (repeatable).

Rules: every submitted response counts; a stereotype is kept if more than half
of all voters answer «Да» («Не уверен(а)» and a skipped item count as not-Да),
so a tie rejects. Each result applies to every template of the item.

Writes (nothing with --check):
- `vote_keep` = yes/no in every row of each worksheet_<Cat>_ru.csv;
- vote_tally.csv: per item, the number of each answer and the verdict
  (aggregate counts only).
Prints the rejected stereotypes and templates and Fleiss' kappa.
"""
import argparse
import csv
import io
import re
import sys
from collections import Counter, defaultdict

from generate_ru import CATS, HERE, read_csv

YES, NO, UNSURE, SKIPPED = "Да", "Нет", "Не уверен(а)", ""
ANSWERS = (YES, NO, UNSURE, SKIPPED)
CODE_COLUMN = "Код участника"
CONSENT_COLUMN = "Согласие на участие"
ITEM = re.compile(r"^(S\d\d)\. …(.*)$", re.S)


def keep(answers):
    """More than half of all voters answered «Да»."""
    return 2 * sum(a == YES for a in answers) > len(answers)


def fleiss_kappa(table):
    """Fleiss' kappa; `table` is one Counter of category -> raters per item."""
    n = sum(table[0].values())
    if n < 2 or any(sum(c.values()) != n for c in table):
        raise ValueError("every item needs the same number of raters, at least 2")
    cats = sorted({k for c in table for k in c})
    p = [sum(c[k] for c in table) / (len(table) * n) for k in cats]
    agree = [(sum(v * v for v in c.values()) - n) / (n * (n - 1)) for c in table]
    p_bar, p_e = sum(agree) / len(agree), sum(x * x for x in p)
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else float("nan")


def read_votes(path, renames):
    """[(code, {item id: answer})] for every submitted response."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head = rows[0]
    code_i = next(i for i, h in enumerate(head) if h.startswith(CODE_COLUMN))
    consent_i = next(i for i, h in enumerate(head) if h.startswith(CONSENT_COLUMN))
    items = {}
    for i, h in enumerate(head):
        m = ITEM.match(h)
        if m:
            items[m.group(1)] = (i, m.group(2).strip())
    voters = []
    for r in rows[1:]:
        if not r[consent_i].strip():
            sys.exit(f"a response without consent: {r[0]}")
        code = renames.get(r[code_i].strip(), r[code_i].strip())
        answers = {k: r[i].strip() for k, (i, _) in items.items()}
        unknown = {a for a in answers.values() if a not in ANSWERS}
        if unknown:
            sys.exit(f"{code}: unexpected answers {unknown}")
        voters.append((code, answers))
    codes = Counter(c for c, _ in voters)
    if any(n > 1 for n in codes.values()) or not all(re.fullmatch(r"V\d+", c) for c in codes):
        sys.exit(f"voter codes must be V1, V2, … and unique (use --rename): {sorted(codes)}")
    return voters, {k: text for k, (_, text) in items.items()}


def write_vote_keep(verdict):
    """Set vote_keep in every worksheet row; verdict: {(category, q_id): bool}."""
    for cat in CATS:
        path = HERE / f"worksheet_{cat}_ru.csv"
        raw = path.read_text(encoding="utf-8")
        rows = list(csv.reader(io.StringIO(raw, newline="")))
        col = rows[0].index("vote_keep")
        cat_i, q_i = rows[0].index("category"), rows[0].index("q_id")
        for r in rows[1:]:
            r[col] = "yes" if verdict[(r[cat_i], int(r[q_i]))] else "no"
        out = io.StringIO(newline="")
        csv.writer(out, lineterminator="\r\n" if "\r\n" in raw else "\n").writerows(rows)
        path.write_text(out.getvalue(), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("votes", help="the Google Forms export (.csv)")
    ap.add_argument("--rename", action="append", default=[], metavar="TYPED=CODE",
                    help="voter code typed in the form = the right code (repeatable)")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args()
    renames = dict(r.rsplit("=", 1) for r in args.rename)

    items = read_csv(HERE / "vote_items.csv")
    voters, form_text = read_votes(args.votes, renames)
    if sorted(form_text) != sorted(r["id"] for r in items):
        sys.exit("the form's items differ from vote_items.csv")
    for r in items:
        if form_text[r["id"]] != r["item_ru"]:
            sys.exit(f"{r['id']}: the form's wording differs from vote_items.csv")
    n = len(voters)
    print(f"{n} voters: {', '.join(sorted(c for c, _ in voters))}")
    if n < 3 or n % 2 == 0:
        print("  note: the rules ask for the researcher plus at least 2 others, an odd number")

    verdict, tally, rejected = {}, [], defaultdict(list)
    for r in items:
        answers = [a[r["id"]] for _, a in voters]
        ok = keep(answers)
        counts = Counter(answers)
        for q in r["templates"].split():
            verdict[(r["category"], int(q))] = ok
        tally.append({"id": r["id"], "category": r["category"], "templates": r["templates"],
                      "item_ru": r["item_ru"], "yes": counts[YES], "no": counts[NO],
                      "unsure": counts[UNSURE], "skipped": counts[SKIPPED], "voters": n,
                      "keep": "yes" if ok else "no"})
        if not ok:
            rejected[r["category"]].append(r)

    print(f"\nkept: {sum(t['keep'] == 'yes' for t in tally)} of {len(tally)} stereotypes, "
          f"{sum(verdict.values())} of {len(verdict)} templates")
    for cat in CATS:
        n_t = sum(1 for (c, _) in verdict if c == cat)
        n_k = sum(v for (c, _), v in verdict.items() if c == cat)
        print(f"  {cat}: {n_k} of {n_t} templates kept")
    print("\nrejected stereotypes:")
    for cat in CATS:
        for r in rejected[cat]:
            t = next(t for t in tally if t["id"] == r["id"])
            print(f"  {r['id']} {cat} q{r['templates'].replace(' ', ', q')}: «{r['item_ru']}» "
                  f"(Да {t['yes']}, Нет {t['no']}, Не уверен(а) {t['unsure']}, skipped {t['skipped']})")

    three = [Counter(a[r["id"]] for _, a in voters) for r in items]
    binary = [Counter(a[r["id"]] == YES for _, a in voters) for r in items]
    print(f"\nFleiss' kappa: {fleiss_kappa(three):.3f} (Да / Нет / Не уверен(а)), "
          f"{fleiss_kappa(binary):.3f} (Да / not Да, the keep decision)")

    if args.check:
        print("\n--check: nothing written")
        return
    write_vote_keep(verdict)
    with open(HERE / "vote_tally.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tally[0]))
        w.writeheader()
        w.writerows(tally)
    print("\nwrote vote_keep in the six worksheets and vote_tally.csv")


if __name__ == "__main__":
    main()
