# Instamart Customer Intelligence & Support Copilot

## Overview

A four-part internal tool built on synthetic, fictional data (all amounts in INR):

- **Part A** cleans the raw exports and stores them in a SQLite database, then answers business questions in SQL and pandas.
- **Part B** segments customers with K-means on RFM features and predicts which customers will lapse.
- **Part C** triages support tickets with an offline mock LLM, with privacy redaction, retries, output validation, a token budget and routing that uses the Part B segments.
- **Part D** is a policy question-answering assistant (TF-IDF retrieval) and a refund agent that calls tools, with safe-stopping limits and crash-and-resume.

Everything runs offline. No API key, network call or paid service is used. `generate_data.py` and `mock_llm.py` are the supplied files and are unmodified.

## Reproduce

Run these from the repository root (Python 3.11 or newer).

1. `python -m venv .venv`
2. Activate it: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux)
3. `pip install -r requirements.txt`
4. `python generate_data.py`  (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python part_a_audit.py`
6. `python part_a_clean.py`
7. `python part_a_validate.py`  (prints `ALL CHECKPOINTS PASSED`)
8. `python part_a_sql.py`
9. `python part_a_pandas.py`
10. `python part_b_segments.py`
11. `python part_b_lapse.py`
12. `python part_c_triage.py`
13. `python part_d_rag.py`
14. `python part_d_agent.py`

Scripts must run in this order, because later Parts read files written by earlier ones.

---

## Part A

### Audit of the raw data

| Figure | Value |
|---|---|
| raw orders.csv rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12,309 |
| non-numeric order_amount | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| customers: blank age | 61 |
| customers: age outside 10-100 | 6 |
| distinct raw city spellings | 24 |
| distinct non-blank gender spellings | 10 |
| blank gender | 30 |

the shape, data types and missing-value table into the README. As a check, expect 61 blank ages (5.08%), 30 blank genders (2.5%) and 2,541 blank delivery times (19.42%).

### Cleaning rules and how many values each changed

**Customers**

| Rule | Changed |
|---|---|
| City: strip spaces and title-case (24 spellings become 6) | 18 spellings merged |
| Gender mapped to male / female / unknown (blank becomes unknown) | 10 spellings become 3 values; 30 blanks become unknown |
| Ages outside 10-100 set to missing | 6 |
| Missing ages filled with the median age of the customer's city (groupby + transform) | 67 (61 blank + 6 invalid) |
| signup_date converted to datetime | 1,200 |

**Orders** (in this order)

| Rule | Changed |
|---|---|
| order_ts converted to datetime | all rows |
| Exact duplicate rows removed | 119 |
| For each order_id keep the record with the latest order_ts (drops earlier pending records) | 656 |
| order_amount converted to numeric; unparseable values filled with round(price x quantity x (1 - discount/100), 2) | 112 |
| Negative delivery_minutes set to missing | 105 |
| Missing delivery_minutes of completed orders filled with the median of that order's product category among completed orders (groupby + transform) | 774 |
| Cancelled orders left with missing delivery_minutes (never delivered, so imputing would invent data) | 1,199 left missing |

Result: 12,309 rows (11,110 completed, 1,199 cancelled, no pending); 0 missing order_amount; completed revenue INR 2,998,668.80. Cities after cleaning: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Genders: female 610, male 560, unknown 30. 0 missing ages, all ages between 18 and 60.

### Storage and validation

The clean tables are in `data/clean/*.csv` and in `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `part_a_validate.py` asserts the reference checkpoints and prints `ALL CHECKPOINTS PASSED`.

### SQL results (`sql/queries.sql`)

| Query | Result |
|---|---|
| Q1 WHERE then HAVING | 30 customers |
| Q2 three-table join | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery | 11 products above the average price |
| Q4 LEFT JOIN | 25 customers with no completed order |
| Q5 CTE | 550 customers above the average spend of INR 2,552.06 |
| Q6 ranking windows | top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows have RANK different from ROW_NUMBER |
| Q7 running total | 183 daily rows; final cumulative INR 2,998,668.80 |

When two customers in the same city have exactly the same total spend, RANK gives them the same number while ROW_NUMBER still gives each a different number, so the two columns disagree for tied rows and for the rows after them.

### pandas cross-check

The pivot table with margins has a grand total of INR 2,998,668.80, which equals the SQL total. Completed order_amount per city:

| City | Sum | Mean | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Distribution of completed order_amount: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR 232.20, fences -283.50 and 645.30, with 0 outliers below and 897 above.

Correlations over completed orders (order_amount, discount_pct, quantity, delivery_minutes): the pair with the largest absolute value is order_amount with discount_pct at -0.095 (order_amount with quantity is +0.087).

This is synthetic quick-commerce data: after cleaning, 12,309 orders from 1,200 customers in six cities between April and September 2025, of which 11,110 were completed. Order amounts are strongly right-skewed: the mean is 269.91 but the median is only 135.00, the skewness is 4.61, and the middle half of orders lies between 64.80 and 297.00. There are 897 completed orders above the upper fence of 645.30 and none below the lower fence. These are probably expensive electronics orders rather than errors, while the real anomalies were in the raw data (duplicates, negative delivery times, unreadable amounts), which we cleaned. Relationships between variables are weak. The largest correlation, between order_amount and discount_pct, is only -0.095, and order_amount with quantity is +0.087. A correlation this small does not establish causation, so we cannot say discounts change order size.

---

## Part B

### RFM (snapshot 2025-10-01)

1,175 customers. Recency: median 19, minimum 0, maximum 180. Frequency: mean 9.46, maximum 33. Monetary: mean 2,552.06, maximum 14,766.55.

### K-means (StandardScaler, n_init=10, random_state=42)

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

Final K = 4.

The inertia falls steeply from K=3 (1268.2) to K=4 (743.1) and then flattens (617.4 at K=5), so the elbow is at K=4. K=4 also has the highest silhouette score (0.526), and it gives four segments the business can act on: big spenders, loyal regulars, occasional shoppers and at-risk customers. I therefore chose K=4.

### Segment profile (K=4)

| segment_id | Name | Size | Mean recency | Mean frequency | Mean monetary | Recommended action |
|---|---|---|---|---|---|---|
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Light nudges and category cross-sell to raise frequency |
| 1 | Premium Big Spenders (**Premium**) | 138 | 24.12 | 6.77 | 7,129.70 | VIP service, early access to electronics, fast support |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Subscription / bundle offers to protect repeat habit |
| 3 | Drifting / At-Risk | 182 | 109.04 | 2.20 | 685.99 | Win-back coupon and reminder before they lapse |

Cluster numbers are only labels and carry no ordering. The Premium flag is set for the segment with the highest mean monetary value. PCA on the scaled RFM matrix: explained_variance_ratio_ = 0.634 and 0.218 (total 0.852). Output: `outputs/customer_segments.csv`.

### Lapse problem

Cutoff 2025-09-01. Cohort: 1,156 customers with a completed order before the cutoff; 443 lapsed (38.32%). Split with `train_test_split(test_size=0.2, stratify=y, random_state=42)`: 924 train, 232 test. Baseline (always "not lapsed") on the test set: accuracy 61.6%, lapsed-class F1 0.

Features: recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure in days.

**Leakage controls.** No order dated on or after 2025-09-01 entered any feature. The segment labels from the K-means step were not used as features. Every encoder (OneHotEncoder with drop="first") and scaler (StandardScaler) is inside a scikit-learn Pipeline and was fitted on training rows only; the resampler also saw only training rows. No feature column contains a missing value (checked in code).

Learned mean of frequency in the scaler (training rows): 8.1147. Full-cohort mean of frequency: 8.0285.

They differ because the scaler learns its mean only from the 924 training rows (8.1147), while the full-cohort mean (8.0285) also includes the test rows, and keeping test rows out of the fitting is exactly what prevents leakage.

### Tuning and test results

| Model | Best parameters | Best CV F1 |
|---|---|---|
| Logistic regression | C = 100 | 0.644 |
| Random forest | max_depth = 8, min_samples_leaf = 10 | 0.685 |

Test set, lapsed class:

| Model | Precision | Recall | F1 | Confusion matrix [[TN, FP], [FN, TP]] |
|---|---|---|---|---|
| Logistic regression | 0.615 | 0.629 | 0.622 | [[108, 35], [33, 56]] |
| Random forest | 0.648 | 0.663 | 0.656 | [[111, 32], [30, 59]] |

Selected model: random forest (test F1 0.656, above the 0.55 requirement).

I selected the random forest because it has the better test F1 (0.656 against 0.622) and also the better precision and recall. Missing a customer who is about to lapse (a false negative) loses future revenue, while wrongly targeting a loyal customer (a false positive) only costs a coupon or a message, so recall matters most. The forest finds 59 of 89 lapsing customers, against 56 for logistic regression, with fewer wrongly targeted loyal customers (32 against 35).

### Class-balance experiment

The training set was over-sampled with duplicates of lapsed rows (random_state=42) to 570 per class. Logistic regression on the unchanged test set:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Un-balanced | 0.615 | 0.629 | 0.622 |
| Balanced | 0.592 | 0.865 | 0.703 |

Recall moved most, from 0.629 to 0.865, while precision fell slightly from 0.615 to 0.592. For the retention team this means they would catch many more customers who are about to lapse, at the price of contacting more customers who would have stayed anyway. That is worth it if the retention offer is cheap.

### Interpretation, persistence and retention list

Top five random-forest features: frequency (0.2842), monetary (0.2748), recency (0.1228), average delivery minutes (0.0624), cancellation rate (0.0480).

Logistic regression coefficient signs (positive means more likely to lapse): recency +0.042, frequency -1.627, monetary +0.222.

frequency (importance 0.2842): customers who order often are in a habit and rarely leave, so low frequency is the clearest warning sign.
monetary (0.2748): total spend separates light users from heavy users, so it helps decide how much a retention offer is worth.
recency (0.1228): the longer since the last order, the more likely the customer is drifting away.
average delivery minutes (0.0624): slow deliveries may hurt the experience and push customers away.
cancellation rate (0.0480): customers with many cancelled orders may be frustrated.

Coefficients: recency (+0.042) is a small positive, so longer gaps slightly raise lapse risk. Frequency (-1.627) is strongly negative, so frequent buyers are much less likely to lapse. Monetary (+0.222) is slightly positive once frequency is held fixed, and I would not read much into it because spend and frequency move together.

The selected model is saved to `outputs/lapse_model.joblib`; the reloaded model reproduces the original test predictions exactly. `outputs/retention_list.csv` holds the Premium customers in the test set ranked by predicted lapse probability: 10 customers, of which 0 truly lapsed. This list is illustrative and not a leakage-free evaluation, because the Premium flag is a post-hoc whole-window label.

---

## Part C

### Guardrails and redaction

4 empty tickets (status `skipped_empty`) and 4 long tickets (estimate 734 tokens, above MAX_INPUT_TOKENS = 400; status `flagged_long`) never reach the model. Redaction substitutions across all 164 tickets: 55 emails, 91 phones, 132 order ids, 278 in total; re-applying the patterns to the redacted text finds 0. An assertion in the code checks that no email, phone or order id appears in any prompt sent.

### Prompts, retries and validation

Both prompts contain the word CLASSIFY, list the four categories and require a JSON-only reply. Only the few-shot prompt contains an `EXAMPLES:` line (four labelled examples, including a fraud example). Estimated prompt size: zero-shot 41 tokens, few-shot 137 tokens.

Retries: at most 3 per ticket, wait = 0.05 x 2^n x uniform(0.9, 1.15) seconds, one log line per retry (`outputs/retry_log.txt`). A full run has 19 TransientLLMErrors, all recovered; waits ranged from 0.0451 to 0.0572 seconds.

Validation statuses for a full run over the 156 eligible tickets: 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`.

### Zero-shot versus few-shot

| Run | Classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126 of 147) | 10,804 | 73.5 |
| Few-shot | 147 | 100% (147 of 147) | 25,780 | 175.4 |

Zero-shot confusion table (rows true, columns predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

Few-shot costs about 102 more tokens per classified ticket (175.4 against 73.5), and 25,780 tokens in total against 10,804, but it lifted accuracy from 85.7% to 100%. It removed all 21 zero-shot errors: 11 fraud tickets and 10 technical tickets that had been labelled billing. The extra tokens are worth paying for when a wrong label is costly, for example fraud sent to billing instead of the security team, which must see it within 2 hours. For low-risk, high-volume tickets, zero-shot would be cheaper.

### Budget, routing and output

Budget demo: zero-shot run with TOKEN_BUDGET = 6000 (prompt about 41 estimated tokens): 69 tickets ended `budget_exhausted` (`outputs/triage_budget_demo.csv`).

Final run (few-shot, TOKEN_BUDGET = 100000) wrote `outputs/triage_results.csv` with 164 rows. Routes: support_queue 102, security_team 34, human_review 17, premium_desk 11. Priority: NORMAL 119, HIGH 45.

**Rule-validated versus human review.** Tickets classified by JSON or keyword validation were routed automatically. 17 tickets went to human review: 4 empty, 4 too long and 9 with invalid model output. No guess was made for these.

---

## Part D

### Chunking and retrieval

40-word chunks with a 10-word overlap give 15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2. For the query "refund policy late delivery" the top three chunks all come from `refund_policy.txt`.

TF measures how often a word appears in a chunk, and IDF measures how rare that word is across all chunks, so rare words count for more. A limitation is that TF-IDF only matches exact words, so a question that says “money back” would not match a chunk that says “refund”.

### Evaluation (14 questions)

| Setting | Answerable: expected doc in top 3 | Answerable: answer contains expected fact | Unanswerable: fallback |
|---|---|---|---|
| Overlap 10 | 12/12 | 12/12 | 2/2 |
| Overlap 0 (13 chunks) | 12/12 | 10/12 | 2/2 |

With overlap 0 two questions break: "What delivery time makes a completed order eligible for a full refund?" and "How soon is money returned when a payment fails but is deducted?".

With overlap 0 the sentences were cut at the 40-word boundary. For the full-refund question, the retrieved chunk had only “receives a full refund of the order amount” and the phrase “more than 45 minutes” was in the previous chunk. For the payment question, “within 48 hours” fell into the next chunk. With overlap 10 the last 10 words repeat in the next chunk, so the sentence stays whole.

### Refund agent: seven scenarios

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 997 |
| S2 | REFUND | 135.00 | 5 | 997 |
| S3 | ESCALATE | 1999.00 | 5 | 984 |
| S4 | DENIED | 0.00 | 4 | 685 |
| S5 | DENIED | 0.00 | 3 | 397 |
| S6 | NOT_FOUND | 0.00 | 2 | 112 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,120 |

All refund amounts equal the expected values. Logs: `outputs/agent_runs.jsonl` (the seven runs), `outputs/refund_ledger.jsonl`, checkpoints in `state/`.

### Stop-safely demonstrations (S1)

| Demonstration | Status |
|---|---|
| token_limit = 200 | FALLBACK_TOKEN_LIMIT |
| time_limit_seconds = 0 | FALLBACK_TIMEOUT |

### Crash and resume (S1)

A simulated crash was raised right after the second tool execution (after its checkpoint was saved), then `resume_agent` continued the run. Tools executed across both parts, in order: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`. Four executions, none repeated. Final status REFUND with 128.25.

### ReAct versus plan-and-execute

In my ReAct agent the model is called after every tool result. S1 needed 5 model calls and 997 tokens, S5 (a cancelled order) 3 calls and 397 tokens, and S6 (an unknown order) only 2 calls and 112 tokens. A plan-and-execute design asks the model once for the whole step list, so a normal refund would need about 2 calls (plan plus final answer) and fewer tokens, because the growing history is not re-sent each time. I did not measure this, so it is an estimate. The weakness is unexpected observations. The plan is written before any tool has run, so for an unknown or cancelled order it still contains steps such as calculate_refund that make no sense; the executor must detect the error and ask the model to replan, which adds a call and more tokens. ReAct simply reads the NOT_FOUND or cancelled result and stops at once. For this refund workflow I would choose ReAct. It costs a few hundred tokens more per request, but money is involved and every decision (order status, delivery time, the INR 1000 escalation limit) depends on what the previous tool returned. Reacting step by step is safer, and it fits the checkpoint and resume design, which saves state after every observation.

---

## Limitations and Next Steps

- The data is synthetic and the language model is a deterministic mock, so accuracy figures (for example 100% few-shot) do not predict real-model performance.
- The Premium flag comes from whole-window RFM, which includes the September window that defines the lapse label. The retention list is therefore illustrative and not a leakage-free evaluation.
- TF-IDF matches words, not meaning, and the policy set is only five short documents.
- The refund ledger has no idempotency key; a replayed run could append a duplicate line. The checkpoint design avoids repeating tools on resume, but a production system should also de-duplicate on order id.
- Use a time-based split with rolling cutoffs for lapse prediction. Replace TF-IDF with an embedding model so paraphrased questions match. Add a de-duplication key on the refund ledger. Add a dashboard for human review of the flagged tickets.
