"""Write REVIEW_SHEET.md: the Russian next to the English, for reading.

One block per template x BBQ version x set: the main set, the control set with
two men, the control set with two women, and (Gender_identity) the masc set.
Each block shows one disambiguated context, split into its ambiguous and
disambiguating parts, with both questions, the answer options and the gold
answer. The main set shows the first value pair in the English data for that
version; the other values fill the same slots.

Usage (from the fork root):
  python my_research/templates/make_review_sheet.py            # writes docs/REVIEW_SHEET.md
  python my_research/templates/make_review_sheet.py --category Age --out /tmp/x.md
"""
import argparse
import json
from collections import defaultdict
from datetime import date

import generate_ru as g

KINDS = {"main": "Main", "control": "Control", "masc": "Masc"}


def blocks(cat, kind, templates, lex, unknowns):
    """{(q, version): [block, ...]} for one file, in data order."""
    control, masc = kind == "control", kind == "masc"
    src = g.ROOT / "data" / f"{cat}{'_control' if control else ''}_en.jsonl"
    samples = [json.loads(line) for line in open(src, encoding="utf-8")]
    problems = g.Problems()
    names = g.control_names(samples, lex, cat, problems) if control else None
    groups = defaultdict(dict)  # (q, version, ru context) -> {polarity: (en, ru)}
    meta = {}
    for s in samples:
        if s["context_condition"] != "disambig":
            continue
        q = int(s["question_index"])
        row = g.pick_row(templates.get(cat, {}).get(q, {}), s, control, masc)
        if row is None:
            continue
        pair = names and names[s["example_id"]]
        ru = g.convert(s, row, lex, unknowns, pair, problems)
        if ru is None:
            continue
        key = (q, s["additional_metadata"]["version"], ru["context"])
        if key not in meta:
            meta[key] = split(s, row, ru, lex, pair, cat, q)
        groups[key][s["question_polarity"]] = (s, ru)
    out = defaultdict(list)
    seen = set()
    for key, by_pol in groups.items():
        if "neg" not in by_pol or "nonneg" not in by_pol:
            continue
        q, version, _ = key
        ambig, disambig, genders, row = meta[key]
        if control:
            pair_kind = "two women" if genders == {"f"} else "two men" if genders == {"m"} else "mixed"
        else:
            pair_kind = ""
        tag = (q, version, pair_kind)
        if tag in seen:
            continue
        seen.add(tag)
        out[(q, version)].append(dict(kind=kind, pair=pair_kind, row=row["variant"].strip() or "main",
                                      ambig=ambig, disambig=disambig, items=by_pol))
    return out


def split(s, row, ru, lex, pair, cat, q):
    """The Russian context as (ambiguous part, disambiguating part)."""
    problems = g.Problems()
    persons = [{"pos": int(k[-1]), "key": k, "value": v, "en_answer": s[k]}
               for k, (v, t) in s["answer_info"].items() if t != "unknown"]
    roles = g.assign(s, row, persons, problems, "")
    for p in persons:
        p["entry"] = (lex.entry("*", "*", pair[p["value"]], p["en_answer"], problems) if pair
                      else lex.entry(cat, q, p["value"], p["en_answer"], problems))
    slots = {slot: p["entry"] for slot, p in roles.items()}
    ambig = g.render(row["ambig_ctx"], slots, problems, "")
    assert ru["context"].startswith(ambig), (cat, q)
    return ambig, ru["context"][len(ambig):].strip(), {e["gender"] for e in slots.values()}, row


def gold(item):
    return item[f"ans{item['label']}"]


def options(item):
    opts = [item[f"ans{i}"] for i in range(3)]
    return " · ".join(f"**{o}**" if i == item["label"] else o for i, o in enumerate(opts))


def write_block(lines, b, show_en):
    en_neg, ru_neg = b["items"]["neg"]
    en_pos, ru_pos = b["items"]["nonneg"]
    label = KINDS[b["kind"]] + (f", {b['pair']}" if b["pair"] else "")
    people = " / ".join(f"«{ru_neg['answer_info'][k][0]}»" for k in ("ans0", "ans1", "ans2")
                        if ru_neg["answer_info"][k][1] != "unknown")
    lines.append(f"**{label}** · row: `{b['row']}` · {people}")
    lines.append("")
    if show_en:
        lines.append(f"- EN: {en_neg['context']}")
        lines.append(f"  - Q−: {en_neg['question']} → {gold(en_neg)}")
        lines.append(f"  - Q+: {en_pos['question']} → {gold(en_pos)}")
    lines.append(f"- RU ambiguous: {b['ambig']}")
    lines.append(f"- RU disambiguating: {b['disambig']}")
    lines.append(f"  - Q−: {ru_neg['question']} → **{gold(ru_neg)}**")
    lines.append(f"  - Q+: {ru_pos['question']} → **{gold(ru_pos)}**")
    lines.append(f"  - Options: {options(ru_neg)}")
    lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", action="append", choices=g.CATS, help="repeatable; default: all six")
    ap.add_argument("--out", default=g.HERE.parent / "docs" / "REVIEW_SHEET.md")
    args = ap.parse_args()
    problems = g.Problems()
    templates, source = g.load_templates(problems)
    lex = g.Lexicon(g.HERE / "ru_lexicon.csv")
    unknowns = {r["en"]: r["ru"].strip() for r in g.read_csv(g.HERE / "ru_unknowns.csv")}
    cats = args.category or g.CATS
    body, n_blocks, n_templates = [], 0, 0
    for cat in cats:
        kinds = ["main", "control"] + (["masc"] if cat == "Gender_identity" else [])
        per = defaultdict(list)
        for kind in kinds:
            for key, bl in blocks(cat, kind, templates, lex, unknowns).items():
                per[key] += bl
        body.append(f"## {cat}\n")
        for q in sorted({q for q, _ in per}):
            variants = ", ".join(f"`{v or 'main'}`" for v in templates[cat][q])
            body.append(f"### {cat} q{q}\n")
            body.append(f"- [ ] read · rows: {variants}\n")
            n_templates += 1
            for version in sorted({v for qq, v in per if qq == q}):
                body.append(f"#### q{q}, version {version}\n" if version not in ("", "None") else f"#### q{q} (one version)\n")
                order = {"main": 0, "control": 1, "masc": 2}
                bl = sorted(per[(q, version)], key=lambda b: (order[b["kind"]], b["pair"] != "two men"))
                shown_control_en = False
                for b in bl:
                    show_en = b["kind"] != "control" or not shown_control_en
                    if b["kind"] == "control":
                        shown_control_en = True
                    write_block(body, b, show_en)
                    n_blocks += 1
                if not any(b["kind"] == "control" and b["pair"] == "two women" for b in bl):
                    body.append("*No female pair in this version's control set.*\n")
    head = f"""# Review sheet ({date.today().isoformat()})

Generated by `make_review_sheet.py` from `{source}` and `ru_lexicon.csv`;
regenerate after every edit:
`python my_research/templates/make_review_sheet.py`.

**What it shows:** {n_templates} templates, {n_blocks} blocks. Each block is one
template × BBQ version × set:
- **Main:** the first value pair in the English data for that version. The
  other values fill the same slots; their forms were checked mechanically.
- **Control, two men** and **Control, two women:** the slot gender changes
  agreement, and the PAIR slots switch to the feminine.
- **Masc** (Gender_identity only).

Each block gives the English first (once per version for the control set),
then the Russian context, split into its ambiguous and disambiguating parts.
Q− is the negative question, Q+ the non-negative one. The gold answer is in
bold, in the options too.

**For each block, check:**
1. **Grammar:** agreement, case, word order, punctuation.
2. **Naturalness:** would a native speaker write it?
3. **Gold supported:** the disambiguating part answers both questions as gold says.
4. **No leak:** the ambiguous part and the questions don't show who did what.
5. **Same meaning as the English.**

Tick `- [ ] read` when a template is done. Note problems in the worksheet or
lexicon, then regenerate this sheet.

"""
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(body))
    print(f"wrote {args.out}: {n_templates} templates, {n_blocks} blocks")


if __name__ == "__main__":
    main()
