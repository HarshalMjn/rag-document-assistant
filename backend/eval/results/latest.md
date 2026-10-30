# Evaluation results

Run at 2026-10-30T09:33:25+00:00 with `gemini-embedding-001` embeddings and `openai/gpt-oss-120b` answers; TOP_K=5, MIN_SCORE=0.55. 21 questions (17 answerable) over 3 documents. Token counts are estimates (chars / 4).

## Retrieval per chunking config

| Config | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR | Refused at retrieval | Answerable top score (min / median) | Unanswerable top score (max) | Index s | Index tokens (est.) |
|---|---|---|---|---|---|---|---|---|---|---|
| 800/120 (default) | 66 | 0.706 | 0.941 | 1.0 | 0.819 | 1/4 | 0.6308 / 0.7178 | 0.647 | 5.9 | 10788 |
| 400/60 | 126 | 0.588 | 0.765 | 0.765 | 0.667 | 1/4 | 0.6653 / 0.7439 | 0.6483 | 9.39 | 10458 |
| 1200/200 | 47 | 0.765 | 0.882 | 1.0 | 0.85 | 1/4 | 0.6135 / 0.6943 | 0.6506 | 5.34 | 11013 |

## Rank of the evidence chunk per question

| Question | 800/120 | 400/60 | 1200/200 |
|---|---|---|---|
| leave-01 | 1 | 1 | 1 |
| leave-02 | 1 | miss | 1 |
| leave-03 | 1 | 1 | 1 |
| leave-04 | 1 | 1 | 1 |
| leave-05 | 1 | 1 | 1 |
| leave-06 | 1 | 1 | 1 |
| diabetes-01 | 3 | 1 | 1 |
| diabetes-02 | 1 | 1 | 1 |
| diabetes-03 | 1 | 1 | 4 |
| diabetes-04 | 4 | 3 | 5 |
| diabetes-05 | 1 | miss | 1 |
| diabetes-06 | 2 | 1 | 1 |
| eatwell-01 | 1 | miss | 1 |
| eatwell-02 | 1 | miss | 2 |
| eatwell-03 | 2 | 2 | 1 |
| eatwell-04 | 1 | 1 | 1 |
| eatwell-05 | 3 | 2 | 2 |
| none-01 | not refused | not refused | not refused |
| none-02 | not refused | not refused | not refused |
| none-03 | not refused | not refused | not refused |
| none-04 | refused | refused | refused |

## Answers (800/120)

Facts correct 17/17, citations correct 17/17, fully correct 17/17, refused correctly 4/4, errors 0. 20 LLM calls, mean 5.312 s; about 23504 prompt + 466 answer tokens (est.).

| Question | Correct | Facts | Citation | Cited | Answer |
|---|---|---|---|---|---|
| leave-01 | yes | yes | yes | leave_policy_v2.txt | 20 days |
| leave-02 | yes | yes | yes | leave_policy_v2.txt | You can carry over up to 5 unused days, and they must be used by 31 March of the following year or they will be lost. |
| leave-03 | yes | yes | yes | leave_policy_v2.txt | You need a medical certificate (doctor’s note) if you are absent for more than three consecutive working days. |
| leave-04 | yes | yes | yes | leave_policy_v2.txt | 16 weeks |
| leave-05 | yes | yes | yes | leave_policy_v2.txt | You receive a replacement day off within the next 30 days. |
| leave-06 | yes | yes | yes | leave_policy_v2.txt | You may work remotely for up to 2 days per week. |
| diabetes-01 | yes | yes | yes | diabetes_how_to_take_control.pdf p2 | Losing even 5‑10% of your body weight can noticeably improve your blood glucose levels. |
| diabetes-02 | yes | yes | yes | diabetes_how_to_take_control.pdf p2 | Yes, type 2 diabetes can go into remission. The document says that most people who have been diagnosed recently and lose more than 15 kg of body weight can achi |
| diabetes-03 | yes | yes | yes | diabetes_how_to_take_control.pdf p2 | Half of your plate should be vegetables or salad. |
| diabetes-04 | yes | yes | yes | eatwell_guide_2018.pdf p6, eatwell_guide_2018.pdf p6, diabetes_how_to_take_control.pdf p3 | One source (S1 and S2) says a pint of standard‑strength lager (beer) has about 136 kcal and a 175 ml glass of wine about 135 kcal. Another source (S4) gives a b |
| diabetes-05 | yes | yes | yes | diabetes_how_to_take_control.pdf p4 | People with diabetes should avoid smoked salmon because it is a ready‑to‑eat cold smoked fish that can carry listeria, and they are at higher risk of serious il |
| diabetes-06 | yes | yes | yes | diabetes_how_to_take_control.pdf p5 | The recommendation is 150 minutes of moderate‑intensity activity each week (or 75 minutes of vigorous activity), plus muscle‑strengthening exercises twice a wee |
| eatwell-01 | yes | yes | yes | eatwell_guide_2018.pdf p7 | No more than 20 g of saturated fat per day. |
| eatwell-02 | yes | yes | yes | eatwell_guide_2018.pdf p8, eatwell_guide_2018.pdf p9, eatwell_guide_2018.pdf p8 | Adults should eat no more than 6 g of salt per day. |
| eatwell-03 | yes | yes | yes | eatwell_guide_2018.pdf p9, eatwell_guide_2018.pdf p9 | Yes. In autumn and winter you should consider taking a daily vitamin D supplement of about 10 µg. |
| eatwell-04 | yes | yes | yes | eatwell_guide_2018.pdf p6 | The weekly alcohol limit is no more than 14 units per week for both men and women. |
| eatwell-05 | yes | yes | yes | eatwell_guide_2018.pdf p10, eatwell_guide_2018.pdf p5 | You should aim to reduce your intake to no more than 70 g of red or processed meat per day. |
| none-01 | yes | - | - | - | I don't have enough information in the uploaded documents to answer that. |
| none-02 | yes | - | - | - | I don't have enough information in the uploaded documents to answer that. |
| none-03 | yes | - | - | - | I don't have enough information in the uploaded documents to answer that. |
| none-04 | yes | - | - | - | I don't have enough information in the uploaded documents to answer that. |
