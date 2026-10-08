# Russian MBBQ: translation decisions

Rules fixed before translating, so that all 98 templates are handled the same
way. Change a rule only together with every template it affects, and log the
change at the bottom.

## Source text

- Translate the **MBBQ English** wording (`en_mbbq_disambig_example`,
  `en_q_neg`, `en_q_nonneg` in the worksheets), not raw BBQ. MBBQ localised
  some text, e.g. "Walmart … Uber" → "supermarket … phone",
  "Who lost Jeopardy?" → "Who lost the quiz show?", "quit STEM" → "quit the
  sciences". BBQ's columns give the slot structure and slot vocabulary.
- The Spanish column is a reference for gendered-language choices only. It has
  slot-substitution errors that must not be copied ("un que llevan un montón de
  papeles", "el silla de ruedas", "es ricas", "los consejos Sofía").
- One Russian row per template. BBQ versions (`versions` column) swap the
  groups between story roles, change who is mentioned first, or switch gender;
  the generator takes all of that from the English samples. Where
  `versions_differ` is `yes`, read `bbq_other_versions` and check that the
  Russian row reads correctly with the groups in either role.

## Profession nouns

- Use a feminine form only if it is standard and stylistically neutral:
  учительница, актриса, медсестра.
- Otherwise use the masculine noun with feminine agreement: «врач пришла»,
  «юрист сказала».
- Never use colloquial or pejorative forms: врачиха, директорша, секретарша.
- SES templates: both people in a sample have the same gender, so gender
  cannot stand in for status.

## Questions

In order of preference:

1. **Zero copula / present tense**, no gender marking: «Кто из них
   секретарь?», «Кто из них плохо разбирается в математике?»
2. **Impersonal or oblique**: «Кому из них поручили вести протокол?», «Чья это
   была идея?»
3. **Doublet**, as Spanish did («¿Quién era el secretario o la secretaria?»),
   only when both forms are neutral.
4. **Masculine default** («Кто был…») only as a last resort; mark
   `gender_leak=1` in the worksheet.

## Agreement and gender

- **Ambiguous context and both questions: no single-person agreement.** Use
  plural predicates for the pair («опоздали», «договорились»). Agreement with
  one person would reveal who did what before the model reasons.
- **Disambiguating context: every word that agrees with a person is a slot**
  (`{{NAME2?вёл|вела}}`), even under a masculine head noun, because the
  control set substitutes names of both genders.
- **Agreement slots hold full forms** (`{{NAME2?вёл|вела}}`), never suffixes
  (`вёл{{NAME2?|а}}` gives «вёла»).
- **Gender-neutral people** take «человек» as the head noun («22-летний
  человек»), not invented gendered forms.

## Controlled gender-leak comparison

- For about 20 Gender_identity templates, add a second row with
  `variant=masc` that differs only in using the masculine-default question.
- Generate it as `data/Gender_identity_ru_masc.jsonl` and run it as language
  `ru_masc`. The Bias_A difference between `ru` and `ru_masc` on the same
  templates measures how much Russian morphology alone moves the score.
- Only the main Gender_identity set: control pairs always share a gender, so
  a masculine default gives no differential cue there.

## Lexical diversity

BBQ varies some words (`{{WORD1}}`: "technical terms / jargon"). Pick one good
Russian rendering per template and write it into the template.

## Control-set names

Map each English control name 1:1 to a Russian name of the same gender, from
an official 2022 civil-registry (ЗАГС) top-names list. Source: **TBD (cite
when chosen)**.

## Change log

| Date | Change | Templates affected |
|---|---|---|
