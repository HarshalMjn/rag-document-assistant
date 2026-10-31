# Sample questions

Questions to try in the UI or the API with the documents in this folder, and the answer
each one should get. The same questions, with the evidence each answer must come from,
are used by the evaluation in [`backend/eval/questions.json`](../backend/eval/questions.json).

## Leave policy (`leave_policy_v1.txt`)

| Question | Expected answer |
|---|---|
| How many days of paid annual leave does a full-time employee get? | 15 days |
| How many sick leave days do I get per year? | 10 days |
| When do I need a doctor's note for being off sick? | For an absence longer than 3 consecutive working days |
| How long is the paid leave for the main carer of a newborn baby? | 16 weeks |
| How much parental leave does the secondary caregiver get? | 4 weeks |
| How many public holidays are there each year? | 10 |
| What do I get if I am asked to work on a public holiday? | A replacement day off within the following 30 days |
| How many days a week am I allowed to work from home? | Up to 2 days |
| How many days in advance must I request annual leave? | At least 10 working days |
| Who do I contact with questions about the policy? | hr@brightline.example |

Each answer should cite `leave_policy_v1.txt`.

### Follow-up question (conversation history)

Ask these two in the same conversation:

1. *How many days of annual leave do I get?* → 15 days
2. *Can I carry some of them over?* → up to 5 days, which must be used by 31 March

"them" is resolved from the history: the second question is rewritten to a search query
such as *Can I carry over annual leave days?* before retrieval.

### Document update

Replace `leave_policy_v1.txt` with `leave_policy_v2.txt` (`PUT /documents/{id}` or the
UI's replace button), then ask again:

| Question | Before (v1) | After (v2) |
|---|---|---|
| How many days of paid annual leave does a full-time employee get? | 15 days | 20 days |
| How fast does annual leave build up each month? | 1.25 days | 1.67 days |

The `document_id` stays the same, the version goes from 1 to 2, and citations point to
version 2 only.

## Diabetes guide (`diabetes_how_to_take_control.pdf`)

| Question | Expected answer | Page |
|---|---|---|
| How much weight do I need to lose to noticeably improve my blood glucose? | 5-10% of body weight | 2 |
| Can type 2 diabetes go into remission, and how much weight loss does that take? | Yes, losing more than 15 kg | 2 |
| How much of my plate should be vegetables or salad? | Half | 2 |
| How many calories are in a pint of beer or a glass of wine? | About 170-250 calories | 3 |
| Why should people with diabetes avoid smoked salmon? | Increased risk of listeriosis | 4 |
| How much exercise per week is recommended for someone with type 2 diabetes? | 150 minutes of moderate activity (or 75 minutes vigorous) | 5 |

## Eatwell Guide (`eatwell_guide_2018.pdf`)

| Question | Expected answer | Page |
|---|---|---|
| What is the most saturated fat an average woman should eat in a day? | No more than 20 g | 7 |
| What is the maximum amount of salt an adult should eat per day? | No more than 6 g | 8 |
| Should I take vitamin D in winter, and how much? | Yes, 10 micrograms a day in autumn and winter | 9 |
| What is the weekly alcohol limit for men and women? | No more than 14 units | 6 |
| I eat a lot of red meat. How much should I cut down to each day? | No more than 70 g | 5 |

PDF citations include the page number.

## Questions the documents do not answer

Each of these should get the insufficient-evidence reply with no citations.

| Question | Why it is a useful check |
|---|---|
| What is the bereavement leave policy? | On topic for the leave policy, but not in it |
| Does the company offer dental insurance to employees? | About the company, but not in any document |
| What is the recommended daily iron intake for adults? | About health, but not in either PDF |
| What is the capital city of France? | Unrelated to every document |

## Document deletion

1. Delete the leave policy (`DELETE /documents/{id}` or the UI).
2. Start a new conversation and ask *How many days of annual leave do I get?*
3. The reply is insufficient evidence: deleted documents are never retrieved.

## Bad inputs (`bad_inputs/`)

| File | Expected result |
|---|---|
| `empty.txt` | Rejected with `400`: the file is empty |
| `not_a_pdf.pdf` | Rejected with `400`: not a valid PDF |
| `unsupported_type.csv` | Rejected with `400`: unsupported file type |
| `scanned_page.pdf` | Upload fails with `422`: no text layer, OCR is not supported |

## Suggested order for a demo

1. Upload `leave_policy_v1.txt` and both PDFs.
2. Ask a leave question, then the follow-up.
3. Ask a PDF question and check the page in the citation.
4. Ask an unanswerable question.
5. Replace the leave policy with v2 and ask the annual leave question again.
6. Delete the leave policy and ask again in a new conversation.
7. Upload the files in `bad_inputs/`.
