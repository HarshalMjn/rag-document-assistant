| Metric | A (800/120) | B (400/60) |
|---|---|---|
| Chunks indexed | 7 | 15 |
| Recall@1 | 0.895 | 0.789 |
| Recall@3 | 1.0 | 0.947 |
| Recall@5 | 1.0 | 1.0 |
| MRR | 0.974 | 0.912 |
| Evidence reaches the LLM (score >= 0.55) | 1.0 | 1.0 |
| Evidence chunk score, mean | 0.662 | 0.694 |
| Evidence chunk score, lowest | 0.565 | 0.593 |
| Unanswerable: highest top score | 0.66 | 0.686 |
| Answer accuracy, all questions | 0.909 | 0.909 |
| Answer accuracy, answerable | 0.895 | 0.895 |
| Unanswerable correctly refused | 3/3 | 3/3 |
| Citation correct (answered questions) | 1.0 | 1.0 |
| Citation precision | 1.0 | 0.974 |
| Mean total latency (ms) | 6703 | 4685 |
| Mean LLM generation latency (ms) | 5443 | 3374 |
| Mean retrieval latency (ms) | 1113 | 1162 |
| LLM input tokens per answer (est.) | 1059 | 754 |

### Config A: every question

| id | evidence rank | top score | ok | cited | answer |
|---|---|---|---|---|---|
| q01 | 1 | 0.7011 | yes | employee_handbook.pdf#0 | 15 days |
| q02 | 1 | 0.6597 | yes | employee_handbook.pdf#0 | Up to 5 unused leave days can be carried over, and they must be used before March 31 of the following year or they expire. |
| q03 | 1 | 0.6601 | yes | employee_handbook.pdf#0 | The department head must approve leave requests that are longer than 5 consecutive days. |
| q04 | 1 | 0.6589 | yes | employee_handbook.pdf#0 | Employees receive 10 days of paid sick leave per year. |
| q05 | 1 | 0.6238 | NO | employee_handbook.pdf#0 | Yes. A medical certificate is required for sick leave longer than two consecutive days. |
| q06 | 1 | 0.7121 | yes | employee_handbook.pdf#1 | 2 days per week |
| q07 | 1 | 0.6825 | yes | employee_handbook.pdf#1 | The company provides a one‑time home office allowance of $300 for equipment such as a monitor, keyboard, or chair. |
| q08 | 1 | 0.6917 | yes | employee_handbook.pdf#1 | Remote employees must be reachable on the company chat between 10:00 and 16:00 local time. |
| q09 | 1 | 0.6451 | NO | employee_handbook.pdf#2 | You may book business class for flights that are six hours or longer. |
| q10 | 1 | 0.7365 | yes | employee_handbook.pdf#2 | $50 |
| q11 | 1 | 0.6798 | yes | employee_handbook.pdf#2 | Expense reports must be submitted within 15 days after the trip ends. |
| q12 | 1 | 0.6654 | yes | it_security_policy.txt#0 | The minimum password length is 14 characters. |
| q13 | 2 | 0.6281 | yes | it_security_policy.txt#0 | It must be reported within 1 hour. |
| q14 | 1 | 0.654 | yes | it_security_policy.txt#1 | You must use the VPN whenever you work from a public network, such as a café, hotel, or airport Wi‑Fi. |
| q15 | 1 | 0.678 | yes | benefits_guide.txt#0 | 16 weeks |
| q16 | 1 | 0.6575 | yes | benefits_guide.txt#0 | 4 weeks |
| q17 | 1 | 0.6951 | yes | benefits_guide.txt#1 | $1,000 per calendar year |
| q18 | 2,1 | 0.6749 | yes | benefits_guide.txt#1, employee_handbook.pdf#1 | You can receive a one‑time home office allowance of $300 (with receipt submission within 30 days) and an annual learning budget of $1,000 for courses, books or conferences. |
| q19 | 3,1 | 0.6103 | yes | benefits_guide.txt#1, employee_handbook.pdf#0 | You can carry over up to 5 unused annual leave days to the next year. Unused learning budget does not carry over. |
| q20 | n/a | 0.6603 | yes | - | I don't have enough information in the uploaded documents to answer that. |
| q21 | n/a | 0.5918 | yes | - | I don't have enough information in the uploaded documents to answer that. |
| q22 | n/a | 0.6441 | yes | - | I don't have enough information in the uploaded documents to answer that. |

### Config B: every question

| id | evidence rank | top score | ok | cited | answer |
|---|---|---|---|---|---|
| q01 | 1 | 0.7053 | yes | employee_handbook.pdf#0 | 15 days |
| q02 | 2 | 0.7001 | yes | employee_handbook.pdf#0, employee_handbook.pdf#1 | Up to 5 unused leave days may be carried over, and they must be used before March 31 of the following year or they expire. |
| q03 | 1 | 0.6947 | yes | employee_handbook.pdf#1 | The department head must approve a leave request of more than 5 consecutive days. |
| q04 | 1 | 0.6985 | yes | employee_handbook.pdf#1 | Employees receive 10 days of paid sick leave per year. |
| q05 | 1 | 0.6928 | NO | employee_handbook.pdf#1 | Yes, a medical certificate is required if the sick leave lasts more than two consecutive days. |
| q06 | 1 | 0.7116 | yes | employee_handbook.pdf#2 | 2 days per week |
| q07 | 3 | 0.7047 | yes | employee_handbook.pdf#2 | The company provides a one‑time home office allowance of $300 for equipment. |
| q08 | 1 | 0.6951 | yes | employee_handbook.pdf#2 | Remote employees must be reachable on the company chat between 10:00 and 16:00 local time. |
| q09 | 1 | 0.6481 | NO | employee_handbook.pdf#4 | You may book business class for flights that are six hours or longer. |
| q10 | 1 | 0.7323 | yes | employee_handbook.pdf#4 | $50 |
| q11 | 2 | 0.695 | yes | employee_handbook.pdf#4 | Expense reports must be submitted within 15 days after the trip ends. |
| q12 | 1 | 0.6814 | yes | it_security_policy.txt#0 | 14 characters |
| q13 | 1 | 0.725 | yes | it_security_policy.txt#1 | It must be reported within 1 hour. |
| q14 | 1 | 0.7033 | yes | it_security_policy.txt#2 | You must use the company VPN whenever you work from a public network, such as a café, hotel, or airport Wi‑Fi. |
| q15 | 1 | 0.7636 | yes | benefits_guide.txt#1 | 16 weeks |
| q16 | 1 | 0.7512 | yes | benefits_guide.txt#1 | The secondary caregiver receives 4 weeks of fully paid parental leave. |
| q17 | 1 | 0.7885 | yes | benefits_guide.txt#2 | $1,000 per calendar year |
| q18 | 4,1 | 0.7111 | yes | benefits_guide.txt#2, employee_handbook.pdf#2 | You can receive a one‑time home office allowance of $300 for equipment, and you have an annual learning budget of $1,000 for courses, books and conferences. |
| q19 | 4,1 | 0.6324 | yes | benefits_guide.txt#2, employee_handbook.pdf#0 | Unused annual leave days can be carried over—up to 5 days may be transferred to the next year, but they must be used before March 31 or they expire. Your learning budget does not carry over; any unused amount is lost at year‑end. |
| q20 | n/a | 0.6861 | yes | - | I don't have enough information in the uploaded documents to answer that. |
| q21 | n/a | 0.5871 | yes | - | I don't have enough information in the uploaded documents to answer that. |
| q22 | n/a | 0.6535 | yes | - | I don't have enough information in the uploaded documents to answer that. |
