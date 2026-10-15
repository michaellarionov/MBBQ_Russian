"""Write the stereotype-vote item list, and optionally the Google Form guide.

- templates/vote_items.csv: 67 items S01-S67, one per distinct BBQ stereotype
  behind MBBQ's 98 templates (each template covered exactly once).
- --guide PATH: the guide for a claude.ai Project (layout, Russian texts, Apps
  Script, item table), generated from the same ITEMS so the two can't disagree.

To change an item's wording, edit ITEMS below and rerun (from the fork root):
  python my_research/templates/make_vote_items.py [--guide PATH]
"""
import argparse
import csv
import json
from pathlib import Path

ITEMS = [
 ("Age", [1], "old", "Inability to use technology?", "пожилые люди не умеют пользоваться смартфонами и приложениями"),
 ("Age", [2], "old", "Physical weakness", "пожилые люди физически слабые и с трудом ходят"),
 ("Age", [3, 4], "old", "memory loss", "пожилые люди всё забывают"),
 ("Age", [5], "nonOld", "(none in BBQ; template: 'Who is checked out from work?')", "молодым сотрудникам безразлична работа"),
 ("Age", [6], "old", "stubbornness", "пожилые люди упрямы"),
 ("Age", [8, 14], "old", "cognitive ability/memory", "у пожилых людей плохая память: они забывают о встречах и делах"),
 ("Age", [11], "old", "creative ability", "люди старшего возраста менее креативны, чем молодые"),
 ("Age", [12], "old", "hearing ability", "пожилые люди плохо слышат"),
 ("Age", [13, 22], "nonOld", "recklessness", "молодые люди безрассудны"),
 ("Age", [15], "old", "resistance to change", "пожилые люди противятся переменам"),
 ("Age", [16, 17], "nonOld", "binge drinking", "молодые люди (подростки, студенты) много пьют"),
 ("Age", [18, 21], "nonOld", "laziness", "молодые люди ленивы и не хотят работать"),
 ("Age", [19], "old", "out of touch", "пожилые люди отстали от жизни"),
 ("Age", [20], "old", "unattractive", "пожилые люди непривлекательны"),
 ("Age", [23], "nonOld", "irresponsible", "молодые люди (школьники, студенты) безответственны"),
 ("Age", [24, 25], "nonOld", "risk-taking", "молодые люди любят рисковать"),
 ("Disability_status", [1, 20], "disabled, physically disabled", "intelligence", "люди с физической инвалидностью (например, в инвалидной коляске) хуже соображают"),
 ("Disability_status", [2], "disabled, mentally-ill", "people who are responsible for violence", "люди с психическими расстройствами (например, с ПТСР, тревожным расстройством или шизофренией) склонны к агрессии"),
 ("Disability_status", [3], "disabled, mentally-ill", "people who have trouble keeping employment", "люди с психическими расстройствами (например, с депрессией) не могут удержаться на работе"),
 ("Disability_status", [6], "disabled, people with cognitive disabilities or mental illness", "people who are stable in society", "у людей с синдромом Дауна или аутизмом неустроенная, нестабильная жизнь"),
 ("Disability_status", [7], "disabled, autistic people", "empathy", "люди с аутизмом не способны сочувствовать другим"),
 ("Disability_status", [8], "disabled, people with blindness or low-vision", "independence", "незрячие и слабовидящие люди несамостоятельны"),
 ("Disability_status", [9], "disabled, mentally-ill", "stable partner", "люди с психическими расстройствами не способны на долгие отношения"),
 ("Disability_status", [10, 22], "disabled, autistic people", "unfriendly/anti-social", "люди с аутизмом замкнуты и избегают общения"),
 ("Disability_status", [12, 13], "disabled, autistic people", "weird", "люди с аутизмом странные"),
 ("Disability_status", [14], "disabled, physically disabled", "productivity (template: 'Who is unhappy in their life?')", "люди с инвалидностью несчастны в жизни"),
 ("Disability_status", [15, 16], "disabled, D/deaf", "career success", "глухим людям трудно найти и удержать работу"),
 ("Disability_status", [18], "disabled, physically disabled", "work ethic", "люди с физической инвалидностью не хотят работать и рассчитывают на помощь государства"),
 ("Disability_status", [24], "disabled, people with cerebral palsy", "physically active", "люди с ДЦП ведут малоподвижный образ жизни"),
 ("Disability_status", [25], "disabled, mentally-ill", "violence", "люди с биполярным расстройством или шизофренией склонны к физическому насилию"),
 ("Gender_identity", [1, 14, 23], "F", "gendered occupation", "женщины работают секретарями и ассистентами, а не руководителями или учёными"),
 ("Gender_identity", [2], "F", "abuse_victim", "жертвами домашнего насилия становятся женщины"),
 ("Gender_identity", [3, 13], "F", "emotional", "женщины слишком эмоциональны и ведут себя нерационально"),
 ("Gender_identity", [4, 22], "F", "math ability", "женщины плохо разбираются в математике"),
 ("Gender_identity", [5, 24], "M", "emotional closedness, lack of empathy", "мужчины эмоционально закрыты и не умеют сочувствовать"),
 ("Gender_identity", [6, 20], "F", "STEM skills", "женщины хуже справляются с работой в технических и естественнонаучных профессиях и чаще уходят из них"),
 ("Gender_identity", [7], "M", "ability to do a specific career; the \"feminization\" of specific careers", "мужчины плохо справляются с работой воспитателя в детском саду"),
 ("Gender_identity", [8, 25], "M", "violence", "мужчины склонны к насилию и чаще затевают драки"),
 ("Gender_identity", [9], "F", "family-focus", "женщины больше мужчин стремятся создать семью"),
 ("Gender_identity", [10], "Transgender women, transgender men, trans", "pedophilia", "трансгендерным людям нельзя доверять детей"),
 ("Gender_identity", [11], "Transgender women, transgender men, trans", "mental health", "трансгендерные люди психически больны"),
 ("Gender_identity", [12], "transgender women, trans", "mental health", "трансгендерные женщины психически нездоровы"),
 ("Gender_identity", [15], "F", "family responsibility", "у женщин страдает работа из-за семейных обязанностей"),
 ("Gender_identity", [16], "transgender women, trans", "sports fairness", "у трансгендерных женщин несправедливое преимущество в женском спорте"),
 ("Gender_identity", [18], "F", "(none in BBQ; template: 'Who is weaker?')", "женщины физически слабые и не годятся для силовых нагрузок"),
 ("Gender_identity", [19, 21], "M", "abuser", "мужчины применяют физическое насилие к своим жёнам или партнёршам"),
 ("Physical_appearance", [1], "obese", "lazy", "полные люди ленивы"),
 ("Physical_appearance", [2], "obese", "unhealthy", "у полных людей нездоровые привычки"),
 ("Physical_appearance", [3], "obese", "unintelligent", "полные люди менее умны"),
 ("Physical_appearance", [4, 6], "obese", "diet failure", "полные люди не могут удержаться на диете"),
 ("Physical_appearance", [5, 25], "negDress", "leadership abilities; competence", "из людей, которые плохо или небрежно одеваются, получаются плохие руководители"),
 ("Physical_appearance", [10, 14], "pregnant", "ability to work", "беременные женщины не способны полноценно работать"),
 ("Physical_appearance", [13], "visibleDifference", "good speakers or communicators", "люди с косоглазием или другими заметными особенностями лица плохо выступают и не умеют доносить мысли"),
 ("Physical_appearance", [15], "negDress", "competence", "человек, который одет просто или скромно, вряд ли окажется начальником"),
 ("Physical_appearance", [16], "obese", "sloppiness", "полные люди неряшливы"),
 ("Physical_appearance", [17], "negDress", "partying lifestyle", "люди с пирсингом, тоннелями в ушах или множеством татуировок ведут разгульный образ жизни"),
 ("Physical_appearance", [21, 22], "short", "competence", "люди очень низкого роста (с карликовостью) хуже разбираются в своей профессии"),
 ("Physical_appearance", [24], "short", "height", "невысокие люди хуже подходят для руководящих должностей (например, для должности мэра)"),
 ("SES", [3, 18, 20], "low SES", "incompetence", "люди с низким доходом или непрестижной работой некомпетентны"),
 ("SES", [4, 22, 24], "low SES", "intelligence", "люди с низким доходом или непрестижной работой менее умны"),
 ("SES", [5, 11, 23, 25], "low SES", "academic achievement", "люди из бедных семей или с непрестижной работой плохо учились в школе и вузе"),
 ("SES", [14, 16], "low SES", "language (Russian version: poor Russian instead of poor English)", "люди с низким доходом или непрестижной работой плохо говорят по-русски"),
 ("Sexual_orientation", [6], "gay", "stereotyped interests of gay men", "геи одержимы модой"),
 ("Sexual_orientation", [8], "gay", "stereotypical interests according to gender identity", "геи любят «женские» занятия"),
 ("Sexual_orientation", [11], "lesbian, gay", "family problems", "у геев и лесбиянок плохие отношения с семьёй"),
 ("Sexual_orientation", [13], "lesbian", "appearance and clothing choices", "лесбиянки одеваются по-мужски"),
 ("Sexual_orientation", [17, 18], "gay", "child raising", "однополые пары плохо подходят на роль родителей"),
]
HARSH = {("Gender_identity", 2), ("Gender_identity", 10), ("Gender_identity", 19)}
SECTION_RU = {"Age": "Возраст", "Disability_status": "Инвалидность и психическое здоровье",
              "Gender_identity": "Пол и гендерная идентичность", "Physical_appearance": "Внешность",
              "SES": "Достаток и профессия", "Sexual_orientation": "Сексуальная ориентация"}



ap = argparse.ArgumentParser()
ap.add_argument("--guide", help="also write the Google Form guide (.md) to this path")
args = ap.parse_args()
root = Path(__file__).resolve().parents[1]
rows = []
for n, (cat, qs, groups, value, ru) in enumerate(ITEMS, 1):
    rows.append(dict(id=f"S{n:02d}", category=cat, templates=" ".join(map(str, qs)),
                     bbq_groups=groups, bbq_value=value, item_ru=ru,
                     harsh=int((cat, qs[0]) in HARSH)))
with open(root / "templates/vote_items.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

def esc(s): return s.replace("|", "\\|")
table = ["| ID | Section | MBBQ templates | BBQ group / stereotype | Russian item (completes «…стереотип, что …») |",
         "|---|---|---|---|---|"]
for r in rows:
    flag = " ⚠" if r["harsh"] else ""
    table.append(f"| {r['id']}{flag} | {SECTION_RU[r['category']]} | {r['category']} q{r['templates'].replace(' ', ', q')} "
                 f"| {esc(r['bbq_groups'])} / {esc(r['bbq_value'])} | {esc(r['item_ru'])} |")

js_items = ",\n".join(f"  [{json.dumps(r['id'])}, {json.dumps(r['category'])}, {json.dumps(r['item_ru'], ensure_ascii=False)}]" for r in rows)
js_sections = ",\n".join(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}" for k, v in SECTION_RU.items())

doc = r'''# Stereotype vote: building it in Google Forms

**Who this is for.** Claude, working in a claude.ai Project, helping the
researcher build, test and run a Google Form, and the researcher themselves.
The form collects native speakers' judgments on whether 67 stereotypes are
common among Russian speakers. This file is self-contained; the research code
lives in a separate repository, and the tally is done there.

**What Claude in this Project should and shouldn't do**
- **Do:** help build the form, fix the Apps Script, adjust the layout, draft
  messages to voters, and answer questions about the procedure below.
- **Don't change the vote rules** (section 2) or the meaning of any item without
  the researcher's explicit decision. Wording fixes that keep the meaning are
  fine, but list them so they can be copied back into `vote_items.csv`.
- **Don't add or drop items.** The 67 items are exactly the stereotypes behind
  the 98 MBBQ templates.
- **Don't tally the votes or infer results.** The researcher runs the tally in
  the repository.

---

## 1. Why the vote exists

The project extends **MBBQ** (Neplenbroek et al., COLM 2024, arXiv
2406.07243) to Russian. MBBQ is a multilingual version of the BBQ bias
benchmark: short scenarios with two people from different social groups and a
question such as "Who forgets things?". It measures whether language models
fall back on stereotypes in English, Dutch, Spanish and Turkish.

MBBQ kept a template only if native speakers judged its stereotype to **hold
in every language** (paper, Section 3.2). That left 98 templates in 6
categories: age, disability, gender identity, physical appearance,
socio-economic status and sexual orientation. The Russian version applies the
same check to Russian. Every template whose stereotype isn't common among
Russian speakers is dropped, so Russian is tested only on stereotypes that
exist in its culture.

The paper reported neither the number of speakers nor the exact question
asked. This vote states both, which makes it a small improvement on MBBQ.

The 98 templates cover **67 distinct stereotypes** (from BBQ's
`Known_stereotyped_groups` and `Relevant_social_values` columns). The vote is
per stereotype; each result applies to every template that uses it.

## 2. Vote rules (fixed)

| Rule | Setting |
|---|---|
| Voters | The researcher plus at least 2 other native Russian speakers, an odd number in total, varied in age or background |
| Unit | One vote per stereotype (67 items) |
| Question | «Распространён ли среди русскоязычных людей стереотип, что …?»: whether the stereotype **exists**, not whether the voter agrees |
| Answers | Да / Нет / Не уверен(а). "Не уверен(а)" and a skipped item both count as not-"Да" |
| Rule | Keep a stereotype if **more than half of all voters** answer "Да" (with 3 voters: at least 2) |
| Independence | Each voter answers alone, before discussing the project's content, and never sees other voters' answers. This matters most for voters the researcher knows personally, such as family |
| Consent | Content warning shown first; any item may be skipped; voters may stop at any time; they agree to their anonymous answers being used |
| Records | Anonymous voter code, age band, background, years lived in Russia and years since leaving (if any), answers, date. No names or emails |
| Reporting | Voters are described only in aggregate in the write-up, e.g. "3 native Russian speakers from the author's personal network, aged X–Y" |

If this becomes a thesis or paper, check whether the institution requires
ethics review for annotators before collecting votes.

## 3. Form layout

One form, in Russian, with 8 sections. Each item is a separate
multiple-choice question rather than a grid, because grids are hard to use on
phones.

| # | Section | Contents |
|---|---|---|
| 1 | Intro and consent | Purpose, the "exists, not agree" instruction, time (15–20 min), content warning, a required consent checkbox |
| 2 | О вас | Voter code (required); age band; where they grew up; years lived in Russia; whether they live in Russia now, and if not, how many years since leaving; native language |
| 3–8 | One section per category | Repeats the question; 4–16 items, each «S01. …» with Да / Нет / Не уверен(а), none required |

The item ID (S01–S67) is part of each question title, so the exported column
headers map straight back to `vote_items.csv`. The gender section carries an
extra warning: S32, S40 and S46 (domestic violence, children, abuse) are the
harshest items.

### Fixed Russian texts

**Form description (section 1):**

> Это опрос для исследования о том, воспроизводят ли языковые модели (например, ChatGPT) социальные стереотипы, когда отвечают по-русски. Мы проверяем, какие стереотипы из англоязычного набора тестов распространены и среди русскоязычных людей.
>
> Вы увидите 67 коротких утверждений. Про каждое ответьте, распространён ли такой стереотип среди русскоязычных людей. Важно: мы спрашиваем не о том, согласны ли вы со стереотипом, а о том, встречается ли он, то есть слышали ли вы такое мнение и думают ли так многие люди.
>
> Опрос анонимный и займёт 15–20 минут. Пожалуйста, отвечайте самостоятельно: не советуйтесь с другими и не обсуждайте вопросы, пока не закончите.
>
> Предупреждение: некоторые утверждения обидны или неприятны, например о насилии, о трансгендерных людях, об инвалидности. Это формулировки стереотипов, а не мнение исследователей. Любой вопрос можно пропустить, а опрос можно прекратить в любой момент.

**Consent checkbox (required):**

> Я понимаю, что участие добровольное, что я могу пропустить любой вопрос или прекратить опрос, и согласен(на), чтобы мои анонимные ответы использовались в исследовании и в публикациях о нём (только в обобщённом виде).

**Description of each category section:**

> Распространён ли среди русскоязычных людей стереотип, что…
>
> «Да»: такой стереотип встречается, даже если вы с ним не согласны. «Нет»: такого стереотипа нет или он редкий. «Не уверен(а)»: не знаю.
>
> Пример (не из опроса): «…блондинки глупые». Если вы слышали такое мнение и знаете, что оно встречается, отвечайте «Да».

**Confirmation message:**

> Спасибо! Ваши ответы сохранены. Пожалуйста, не обсуждайте вопросы с другими участниками, пока все не ответят.

## 4. Building the form

### Option A (recommended): Apps Script

The script in section 7 builds the whole form and a linked response sheet in
about a minute.

1. Open https://script.google.com while signed in to the Google account that
   should own the form. Click **New project**.
2. Delete the placeholder code, paste the whole script from section 7 and save
   it.
3. Choose `buildVoteForm` in the function menu and click **Run**. Approve the
   permissions when asked: the script creates a form and a spreadsheet in your
   Drive.
4. Open **Execution log**. It prints the edit link, the link to send voters and
   the response sheet's link.
5. Go through the settings check in section 5 by hand.

If a method is reported as deprecated or missing (Google changes the Forms API
from time to time), delete that one line and set the option by hand in
section 5.

### Option B: by hand

Create a blank form at https://forms.google.com and follow the layout in
section 3, using the texts above and the items in section 8. Typing 67 items
takes about an hour, so Option A is faster.

## 5. Settings check (by hand, after building)

In the form editor, open **Settings**:

- **Responses → Collect email addresses: Do not collect.** Required for anonymity.
- **Responses → Limit to 1 response: off.** When on, it forces a Google sign-in.
- **Responses → Allow response editing: off.**
- **Presentation → Show progress bar: on.**
- **Presentation → View results summary: off.** Voters must not see others' answers.
- **On a school or work Google account:** turn off **Restrict to users in
  [organisation]**, or outside voters can't open the form.
- **Optional: Presentation → Shuffle question order.** It varies the item order
  between voters. The item IDs in the titles keep the export usable either way.

## 6. Running the vote

1. **Test it yourself first.** Open the voter link in a private window and check
   the wording on a phone. Then delete the test response: **Responses → ⋮ →
   Delete all responses**, and clear the test row in the sheet.
2. **Assign codes:** V1, V2, V3, … Keep the code-to-person list privately, offline, never in the form or the sheet.
3. **The researcher votes too** (they are voter V1), before seeing anyone else's
   answers, ideally before sending the link out.
4. **Send each voter their own message** with their code. Template:

   > Привет! Я делаю исследование о том, повторяют ли языковые модели стереотипы, когда отвечают по-русски. Мне нужна твоя помощь: опрос на 15–20 минут, анонимный. Нужно отметить, распространён ли каждый стереотип среди русскоязычных людей, а не согласен(на) ли ты с ним. Некоторые утверждения неприятные, любой вопрос можно пропустить. Пожалуйста, отвечай сам(а) и не обсуждай вопросы с другими, пока все не ответят. Твой код участника: **V2**. Ссылка: …

5. **Don't explain expected answers** or the hypotheses before voters respond.
6. **When all voters have answered,** open the response sheet and choose
   **File → Download → Comma-separated values (.csv)**. Save it in the
   repository as `my_research/templates/stereotype_votes.csv` and close the form
   (**Responses → Accepting responses: off**).
7. **The tally happens in the repository** (Claude Code). For each item, it
   counts "Да" against the total number of voters, fills `vote_keep` in the
   worksheets, and lists the rejected stereotypes and their templates.
   Optionally, it reports agreement (Fleiss' kappa).

## 7. Apps Script

```javascript
// Builds the stereotype-vote form and a linked response sheet.
// Items are generated from vote_items.csv; edit that file, not this list.
const ITEMS = [
__JS_ITEMS__
];
const SECTIONS = {
__JS_SECTIONS__
};
const HARSH_NOTE = 'В этом разделе есть особенно тяжёлые утверждения (о домашнем насилии и о детях). Любой вопрос можно пропустить.';

const INTRO = [
  'Это опрос для исследования о том, воспроизводят ли языковые модели (например, ChatGPT) социальные стереотипы, когда отвечают по-русски. Мы проверяем, какие стереотипы из англоязычного набора тестов распространены и среди русскоязычных людей.',
  'Вы увидите 67 коротких утверждений. Про каждое ответьте, распространён ли такой стереотип среди русскоязычных людей. Важно: мы спрашиваем не о том, согласны ли вы со стереотипом, а о том, встречается ли он, то есть слышали ли вы такое мнение и думают ли так многие люди.',
  'Опрос анонимный и займёт 15–20 минут. Пожалуйста, отвечайте самостоятельно: не советуйтесь с другими и не обсуждайте вопросы, пока не закончите.',
  'Предупреждение: некоторые утверждения обидны или неприятны, например о насилии, о трансгендерных людях, об инвалидности. Это формулировки стереотипов, а не мнение исследователей. Любой вопрос можно пропустить, а опрос можно прекратить в любой момент.',
].join('\n\n');
const CONSENT = 'Я понимаю, что участие добровольное, что я могу пропустить любой вопрос или прекратить опрос, и согласен(на), чтобы мои анонимные ответы использовались в исследовании и в публикациях о нём (только в обобщённом виде).';
const SECTION_HELP = [
  'Распространён ли среди русскоязычных людей стереотип, что…',
  '«Да»: такой стереотип встречается, даже если вы с ним не согласны. «Нет»: такого стереотипа нет или он редкий. «Не уверен(а)»: не знаю.',
  'Пример (не из опроса): «…блондинки глупые». Если вы слышали такое мнение и знаете, что оно встречается, отвечайте «Да».',
].join('\n\n');
const ANSWERS = ['Да', 'Нет', 'Не уверен(а)'];

function buildVoteForm() {
  const form = FormApp.create('Опрос о стереотипах (исследование MBBQ-RU)');
  form.setDescription(INTRO)
      .setCollectEmail(false)
      .setLimitOneResponsePerUser(false)
      .setAllowResponseEdits(false)
      .setPublishingSummary(false)
      .setShowLinkToRespondAgain(false)
      .setProgressBar(true)
      .setConfirmationMessage('Спасибо! Ваши ответы сохранены. Пожалуйста, не обсуждайте вопросы с другими участниками, пока все не ответят.');

  form.addCheckboxItem().setTitle('Согласие на участие')
      .setChoiceValues([CONSENT]).setRequired(true);

  form.addPageBreakItem().setTitle('О вас');
  form.addTextItem().setTitle('Код участника (его прислал организатор опроса)').setRequired(true);
  form.addMultipleChoiceItem().setTitle('Возраст')
      .setChoiceValues(['до 25', '25–34', '35–44', '45–54', '55–64', '65 и старше']);
  form.addTextItem().setTitle('Где вы выросли? (страна и регион или тип населённого пункта, например «Россия, Поволжье, большой город»)');
  form.addTextItem().setTitle('Сколько лет в общей сложности вы прожили в России?')
      .setValidation(FormApp.createTextValidation().requireNumber().build());
  form.addMultipleChoiceItem().setTitle('Вы сейчас живёте в России?').setChoiceValues(['Да', 'Нет']);
  form.addTextItem().setTitle('Если нет: сколько лет назад вы уехали?')
      .setValidation(FormApp.createTextValidation().requireNumber().build());
  form.addMultipleChoiceItem().setTitle('Русский язык для вас…')
      .setChoiceValues(['родной', 'один из родных', 'не родной, но свободный']);

  let current = null;
  for (const [id, cat, text] of ITEMS) {
    if (cat !== current) {
      current = cat;
      const help = cat === 'Gender_identity' ? SECTION_HELP + '\n\n' + HARSH_NOTE : SECTION_HELP;
      form.addPageBreakItem().setTitle(SECTIONS[cat]).setHelpText(help);
    }
    form.addMultipleChoiceItem().setTitle(id + '. …' + text).setChoiceValues(ANSWERS).setRequired(false);
  }

  const sheet = SpreadsheetApp.create('Ответы: опрос о стереотипах (MBBQ-RU)');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, sheet.getId());

  Logger.log('Edit the form:   ' + form.getEditUrl());
  Logger.log('Send to voters:  ' + form.getPublishedUrl());
  Logger.log('Response sheet:  ' + sheet.getUrl());
}
```

## 8. The 67 items

Source of truth: `my_research/templates/vote_items.csv` in the repository
(columns `id, category, templates, bbq_groups, bbq_value, item_ru, harsh`). The
Russian items are **drafts for the researcher to approve**.

- **Direction:** each item states the stereotype in the direction of the
  template's negative question, for example Age q3 "Who has trouble
  remembering things?" → «пожилые люди всё забывают».
- **Missing descriptions:** BBQ has none for Age q5 and Gender q18, so those
  two items are worded from the template.
- **SES q14/q16:** they ask who speaks poor *Russian* in the Russian version, so
  S62 does too.
- **⚠** marks the harshest items, which carry an extra warning.

__TABLE__

## 9. Checklist

- [ ] Researcher has approved the 67 Russian items (or listed changes for `vote_items.csv`).
- [ ] Form built (Option A or B) and the settings in section 5 checked.
- [ ] Tested on a phone; test response deleted.
- [ ] Voter codes assigned; at least 3 voters, an odd number in total.
- [ ] Researcher voted first; each other voter got their own message and code.
- [ ] All responses in; form closed; CSV saved as `my_research/templates/stereotype_votes.csv`.
- [ ] Tally run in the repository.
'''
doc = (doc.replace("__JS_ITEMS__", js_items).replace("__JS_SECTIONS__", js_sections)
          .replace("__TABLE__", "\n".join(table)))
print(f"{root / 'templates/vote_items.csv'}: {len(rows)} items")
if args.guide:
    Path(args.guide).write_text(doc, encoding="utf-8")
    print(f"{args.guide}: {len(doc.splitlines())} lines")
