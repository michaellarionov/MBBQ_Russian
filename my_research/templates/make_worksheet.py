"""Build one translation worksheet per category for the templates MBBQ kept.

Each row joins BBQ's original template (slot structure, slot vocabulary) with
one rendered English MBBQ sample of the same version (MBBQ localised some
wording) and the parallel Spanish sample (a gendered-language reference).
Russian columns start empty and become ru_templates.csv once filled in.

Most templates have 2-4 BBQ versions. They tell the same story with the
groups swapped between roles, the mention order changed, or the other gender;
the generator handles those from the English samples, so one Russian row per
template is enough. The worksheet shows version a; `versions_differ` marks
templates whose other versions are also worded differently, with that wording
in `bbq_other_versions`.

Usage (from the fork root, with nyu-mll/BBQ cloned next to the fork):
  python my_research/templates/make_worksheet.py --bbq ../BBQ
"""
import argparse
import csv
import json
import re
from pathlib import Path

CATS = ["Gender_identity", "Disability_status", "Physical_appearance",
        "Age", "SES", "Sexual_orientation"]  # suggested work order
TEXT = {"bbq_ambig": "Ambiguous_Context", "bbq_disambig": "Disambiguating_Context",
        "bbq_q_neg": "Question_negative_stereotype", "bbq_q_nonneg": "Question_non_negative",
        "bbq_answer_neg": "Answer_negative"}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def pick(samples, version, condition, polarity):
    """First sample of this condition and polarity, preferring the version.

    Templates without BBQ versions (Sexual_orientation) are "None" in MBBQ.
    """
    match = [s for s in samples if s["context_condition"] == condition
             and s["question_polarity"] == polarity]
    same = [s for s in match if s["additional_metadata"].get("version", "") == version]
    return (same or match or [{}])[0]


def neg_slot(answer_negative):
    m = re.search(r"\{\{(NAME[12])\}\}", answer_negative)
    return m.group(1) if m else "CHECK"


def normalise(text):
    """Template text with slot order and gendered words ignored."""
    text = re.sub(r"\{\{NAME[12]\}\}", "{{N}}", text)
    text = re.sub(r"\b(she|he|her|his|him|herself|himself|woman|man|women|men|girl|boy)\b",
                  "<g>", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbq", default="../BBQ")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="my_research/templates")
    args = ap.parse_args()
    for cat in CATS:
        en = load_jsonl(Path(args.data) / f"{cat}_en.jsonl")
        es = {s["example_id"]: s for s in load_jsonl(Path(args.data) / f"{cat}_es.jsonl")}
        with open(Path(args.bbq) / "templates" / f"new_templates - {cat}.csv",
                  encoding="utf-8") as f:
            bbq = {}
            for r in csv.DictReader(f):
                if r["Q_id"].strip():
                    bbq.setdefault(int(r["Q_id"]), []).append(r)
        by_q = {}
        for s in en:
            by_q.setdefault(int(s["question_index"]), []).append(s)
        rows = []
        for q in sorted(by_q):
            versions = bbq[q]
            b = versions[0]  # version a
            v = b.get("version", "").strip()
            samples = by_q[q]
            dis = pick(samples, v, "disambig", "neg") or samples[0]
            nonneg = pick(samples, v, "disambig", "nonneg")
            s_dis = es.get(dis.get("example_id"), {})
            differ = len({tuple(normalise(r[c]) for c in TEXT.values()) for r in versions}) > 1
            other = []
            if differ:
                for r in versions[1:]:
                    for col, c in TEXT.items():
                        if r[c] != b[c]:
                            other.append(f"[{r.get('version', '').strip() or '?'}] {col}: {r[c].strip()}")
            rows.append({
                "category": cat, "q_id": q,
                "versions": ",".join(r.get("version", "").strip() or "?" for r in versions),
                "versions_differ": "yes" if differ else "",
                **{col: b[c].strip() for col, c in TEXT.items()},
                "bbq_names": b.get("Names", ""),
                "bbq_lexical_diversity": b.get("Lexical_diversity", ""),
                "bbq_other_versions": "\n".join(other),
                "en_mbbq_disambig_example": dis.get("context", ""),
                "en_q_neg": dis.get("question", ""),
                "en_q_nonneg": nonneg.get("question", ""),
                "es_mbbq_disambig_example": s_dis.get("context", ""),
                "es_q_neg": s_dis.get("question", ""),
                # Russian columns to fill (copy to ru_templates.csv when done)
                "vote_keep": "", "variant": "",
                "neg_answer_slot": neg_slot(b["Answer_negative"]),
                "ambig_ctx": "", "disambig_ctx": "", "q_neg": "", "q_nonneg": "",
                "gender_leak": "", "notes": "",
            })
        out = Path(args.out) / f"worksheet_{cat}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        n_differ = sum(r["versions_differ"] == "yes" for r in rows)
        unresolved = [r["q_id"] for r in rows if r["neg_answer_slot"] == "CHECK"]
        print(f"{out}: {len(rows)} templates, {n_differ} with differently worded "
              f"versions, unresolved neg_answer_slot: {unresolved or 'none'}")


if __name__ == "__main__":
    main()
