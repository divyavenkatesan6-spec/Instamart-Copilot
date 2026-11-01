"""Part C: support-ticket triage with guardrails (input checks, PII redaction, retries, output validation,
token budget and routing using the Part B segments). All model calls go through the offline mock_llm.py."""
import json
import os
import random
import re
import time
import pandas as pd
import mock_llm
from mock_llm import est_tokens, TransientLLMError

os.makedirs("outputs", exist_ok=True)
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED = ["billing", "delivery", "technical", "fraud"]
random.seed(42)

tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False, dtype=str)   # empty text stays ""
segments = pd.read_csv("outputs/customer_segments.csv")
seg_lookup = segments.set_index("customer_id")[["segment_name", "is_premium"]].to_dict("index")

# ---------------- Task 2: redaction ----------------
PATTERNS = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
            ("PHONE", re.compile(r"\b[6-9]\d{9}\b")),
            ("ORDER_ID", re.compile(r"ORD-\d{6}"))]
def redact(text, counts=None):
    for name, pat in PATTERNS:
        text, n = pat.subn(f"[{name}]", text)
        if counts is not None:
            counts[name] += n
    return text

# ---------------- Task 3: prompts ----------------
FORMAT = ('Reply with JSON only, in exactly this form: {"category": "...", "reason": "..."}. '
          'No other text.')
ZERO_SHOT = ("You CLASSIFY customer support tickets for a grocery delivery app. "
             "Allowed categories: billing, delivery, technical, fraud. " + FORMAT)
FEW_SHOT = ZERO_SHOT + "\nEXAMPLES:\n" \
    "Ticket: I was charged twice for order [ORDER_ID]. Please refund the extra payment.\n" \
    '{"category": "billing", "reason": "double charge"}\n' \
    "Ticket: My order [ORDER_ID] has not arrived and the rider is not picking up calls.\n" \
    '{"category": "delivery", "reason": "order not received"}\n' \
    "Ticket: The app crashes every time I open the cart.\n" \
    '{"category": "technical", "reason": "app crash"}\n' \
    "Ticket: There is an unauthorized transaction on my card. I did not place it.\n" \
    '{"category": "fraud", "reason": "unauthorized transaction"}'
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "EXAMPLES:" in FEW_SHOT

# ---------------- Task 4: resilient call ----------------
retry_log = []
def call_with_retry(messages, ticket_id):
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            if n == MAX_RETRIES:
                raise
            retry_no = n    # retry number n = 0, 1, 2 (counted per ticket: first failure -> retry 0)
            wait = BASE_DELAY * 2 ** retry_no * random.uniform(0.9, 1.15)
            retry_log.append(f"ticket={ticket_id} retry={retry_no} wait={wait:.4f}s")
            time.sleep(wait)

# ---------------- Task 5: validation ----------------
def validate(content):
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (content or "").lower()))
        found = [c for c in ALLOWED if c in words]
        return ("classified_keyword", found[0]) if len(found) == 1 else ("flagged_invalid_output", "")
    if isinstance(obj, dict) and obj.get("category") in ALLOWED:
        return "classified_json", obj["category"]
    return "flagged_invalid_output", ""        # valid JSON but wrong category -> human review, never guess

# ---------------- pipeline ----------------
def run_pipeline(system_prompt, token_budget=None):
    mock_llm.reset()
    retry_log.clear()
    total_tokens, rows = 0, []
    for _, t in tickets.iterrows():                       # file order
        text, status, category, used = t["text"], "", "", 0
        if text.strip() == "":
            status = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            status = "budget_exhausted"
        else:
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": redact(text)}]
            for pat_name, pat in PATTERNS:                # privacy guardrail check on what is actually sent
                assert not any(pat.search(m["content"]) for m in msgs if m["role"] == "user"), "PII reached prompt"
            resp = call_with_retry(msgs, t["ticket_id"])
            used = resp.usage.total_tokens
            total_tokens += used
            status, category = validate(resp.choices[0].message.content)
        rows.append({"ticket_id": t["ticket_id"], "customer_id": t["customer_id"], "status": status,
                     "category": category, "tokens_used": used, "true_category": t["true_category"]})
    return pd.DataFrame(rows), total_tokens, list(retry_log)

def route(df):
    seg_names, prios, routes = [], [], []
    for _, r in df.iterrows():
        info = seg_lookup.get(r["customer_id"])
        seg_names.append(info["segment_name"] if info else "")
        premium = bool(info["is_premium"]) if info else False
        if r["status"] in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
            routes.append("human_review"); prios.append("NORMAL")
        elif r["category"] == "fraud":
            routes.append("security_team"); prios.append("HIGH")
        elif premium:
            routes.append("premium_desk"); prios.append("HIGH")
        else:
            routes.append("support_queue"); prios.append("NORMAL")
    out = df.copy()
    out["segment_name"], out["priority"], out["route"] = seg_names, prios, routes
    return out[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]

# ================= REPORT =================
if __name__ == "__main__":
    # Task 1
    empty = (tickets["text"].str.strip() == "").sum()
    long_ = tickets["text"].apply(lambda s: est_tokens(s) > MAX_INPUT_TOKENS).sum()
    long_tok = tickets.loc[tickets["text"].apply(lambda s: est_tokens(s) > MAX_INPUT_TOKENS), "text"].apply(est_tokens).unique()
    print(f"Task 1: empty tickets={empty} (ref 4), long tickets={long_} (ref 4), long estimate={[int(x) for x in long_tok]} (ref 734)")

    # Task 2
    counts = {"EMAIL": 0, "PHONE": 0, "ORDER_ID": 0}
    redacted = [redact(t, counts) for t in tickets["text"]]
    left = sum(len(p.findall(t)) for t in redacted for _, p in PATTERNS)
    print(f"Task 2: emails={counts['EMAIL']} phones={counts['PHONE']} order_ids={counts['ORDER_ID']} "
          f"total={sum(counts.values())} (ref 55 / 91 / 132 / 278); left after redaction={left} (ref 0)")

    # Tasks 4-6
    results = {}
    for name, prompt in [("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)]:
        df, tot, log = run_pipeline(prompt)
        results[name] = (df, tot, log)
        st = df["status"].value_counts().to_dict()
        ok = df[df["status"].isin(["classified_json", "classified_keyword"])]
        acc = (ok["category"] == ok["true_category"]).mean() * 100
        waits = [float(l.split("wait=")[1][:-1]) for l in log]
        print(f"\n== {name} ==  prompt tokens (est)={est_tokens(prompt)}")
        print(f"status counts: {st}  (ref 135 json / 12 keyword / 9 invalid)")
        print(f"transient errors recovered by retry: {len(log)} (ref 19); wait range {min(waits):.4f}-{max(waits):.4f}s (expect 0.045-0.0575)")
        print(f"classified={len(ok)} (ref 147) accuracy={acc:.1f}% ({int((ok['category']==ok['true_category']).sum())} of {len(ok)})")
        print(f"total tokens={tot} tokens per classified ticket={tot/len(ok):.1f}")
        if name == "zero-shot":
            print("Confusion table (rows = true, columns = predicted):")
            print(pd.crosstab(ok["true_category"], ok["category"]).to_string())
    with open("outputs/retry_log.txt", "w") as f:
        f.write("\n".join(results["few-shot"][2]) + "\n")

    # Task 7a: budget demo
    BUDGET = 6000
    df_b, tot_b, _ = run_pipeline(ZERO_SHOT, token_budget=BUDGET)
    n_ex = int((df_b["status"] == "budget_exhausted").sum())
    print(f"\nBudget demo: TOKEN_BUDGET={BUDGET} -> budget_exhausted tickets={n_ex} (need >= 1); tokens spent={tot_b}")
    route(df_b).to_csv("outputs/triage_budget_demo.csv", index=False)
    assert n_ex >= 1, "lower the demo budget"

    # Task 7b: final few-shot run with routing
    df_f, tot_f, _ = run_pipeline(FEW_SHOT, token_budget=100000)
    final = route(df_f)
    final.to_csv("outputs/triage_results.csv", index=False)
    print(f"\nFinal triage_results.csv rows={len(final)} (ref 164)")
    print("Route counts:", final["route"].value_counts().to_dict())
    print("Priority counts:", final["priority"].value_counts().to_dict())
