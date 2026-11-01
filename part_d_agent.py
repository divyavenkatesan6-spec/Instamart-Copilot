"""Part D, tasks 5-8: refund agent (tools, ReAct loop, safe stopping, checkpoint/resume).
Run `python part_d_agent.py` to execute the 7 scenarios and the three demonstrations."""
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
import pandas as pd
import mock_llm
from part_d_rag import retrieve

DB = os.path.join("data", "instamart.db")
LEDGER = os.path.join("outputs", "refund_ledger.jsonl")
RUNS = os.path.join("outputs", "agent_runs.jsonl")
DEMOS = os.path.join("outputs", "agent_demos.jsonl")
STATE_DIR = "state"
os.makedirs("outputs", exist_ok=True); os.makedirs(STATE_DIR, exist_ok=True)
CURRENT = {"run_id": ""}          # set by the loop so ledger lines can name the run that wrote them

# ---------------- Task 5: tools ----------------
def get_order(order_id):
    con = sqlite3.connect(DB)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                      (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}

def search_policy(query):
    return {"results": [{"source": h["source"], "score": h["score"], "text": h["text"]} for h in retrieve(query, 3)]}

def calculate_refund(order_amount, delivery_minutes):
    """Delivered in more than 45 minutes -> 100% of the order amount; otherwise 0."""
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}

def _ledger(tool, order_id, refund_inr):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool, "run_id": CURRENT["run_id"],
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")

def issue_refund(order_id, refund_inr):
    _ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}

def escalate_to_human(order_id, refund_inr):
    _ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}

def _spec(name, desc, props):
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}
TOOLS_SPEC = [
    _spec("get_order", "Look up an order by id in the database.", {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the company policy documents.", {"query": {"type": "string"}}),
    _spec("calculate_refund", "Compute the refund for an order from its amount and delivery time.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue an automatic refund of up to INR 1000.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

AGENT_SYSTEM = ("You are a refund AGENT. Look up the order, check the policy, calculate the refund, "
                "then issue it or escalate it. Finish with STATUS: explanation.")

class SimulatedCrash(Exception):
    pass

# ---------------- Tasks 6-8: loop, limits, checkpoints ----------------
def _checkpoint(st):
    with open(os.path.join(STATE_DIR, f"{st['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, default=str)

def _fallback(st, status, reason):
    tools = ", ".join(s["tool"] for s in st["steps"]) or "none"
    answer = (f"HANDOFF to human agent: automated run stopped ({reason}) after {len(st['steps'])} steps "
              f"[{tools}]; please review the request manually.")
    return _result(st, status, answer)

def _result(st, status, final_answer):
    return {"run_id": st["run_id"], "scenario_id": st["scenario_id"], "status": status, "final_answer": final_answer,
            "steps": st["steps"], "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"],
            "tools_executed": st["tools_executed"]}

def _loop(st, cfg, crash_after_tools=None):
    start = time.monotonic()
    while True:
        if time.monotonic() - start >= cfg["time_limit_seconds"]:                      # (c) wall-clock limit
            return _fallback(st, "FALLBACK_TIMEOUT", f"time limit {cfg['time_limit_seconds']}s reached")
        resp = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > cfg["token_limit"]:                                     # (b) token limit, tool NOT executed
            return _fallback(st, "FALLBACK_TOKEN_LIMIT", f"token limit {cfg['token_limit']} exceeded")
        msg = resp.choices[0].message
        if not msg.tool_calls:                                                         # final answer
            text = msg.content or ""
            return _result(st, text.split(":")[0].strip(), text)
        if st["llm_calls"] >= cfg["max_iterations"]:                                   # (a) iteration limit
            return _fallback(st, "FALLBACK_MAX_ITERATIONS", f"{cfg['max_iterations']} model calls without a final answer")
        tc = msg.tool_calls[0]
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        CURRENT["run_id"] = st["run_id"]
        fn = TOOL_MAP.get(tc.name)
        obs = fn(**tc.arguments) if fn else {"error": f"unknown tool {tc.name}"}
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
        st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        st["tools_executed"].append(tc.name)
        st["step"] = len(st["steps"])
        _checkpoint(st)                                                                # after every observation
        if crash_after_tools and len(st["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools}")

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    mock_llm.reset()
    st = {"run_id": run_id or f"{scenario_id}_{uuid.uuid4().hex[:6]}", "scenario_id": scenario_id,
          "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "llm_calls": 0, "step": 0, "tools_executed": [], "steps": [],
          "config": {"max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}}
    return _loop(st, st["config"], crash_after_tools)

def resume_agent(run_id):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        st = json.load(f)
    mock_llm.reset()
    return _loop(st, st["config"])

# ---------------- runner ----------------
def refund_amount(res):
    for s in reversed(res["steps"]):
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return s["observation"]["refund_inr"]
    return 0.0

def log_run(path, res):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: res[k] for k in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps")}) + "\n")

if __name__ == "__main__":
    for p in (LEDGER, RUNS, DEMOS):
        open(p, "w").close()                              # fresh outputs for a reproducible run
    for fn in os.listdir(STATE_DIR):
        os.remove(os.path.join(STATE_DIR, fn))

    sc = pd.read_csv("data/raw/agent_scenarios.csv")
    print("scenario | status | refund_inr | model_calls | tokens_used | expected")
    s1_msg = None
    for _, r in sc.iterrows():
        res = run_agent(r["message"], r["scenario_id"], run_id=f"{r['scenario_id']}_main")
        log_run(RUNS, res)
        amt = refund_amount(res)
        ok = res["status"].startswith(r["expected_outcome"]) and abs(amt - float(r["expected_refund_inr"])) <= 0.01
        print(f"{r['scenario_id']} | {res['status']} | {amt:.2f} | {res['llm_calls']} | {res['tokens_used']} | "
              f"{r['expected_outcome']} {float(r['expected_refund_inr']):.2f} {'OK' if ok else 'MISMATCH'}")
        if r["scenario_id"] == "S1":
            s1_msg = r["message"]

    print("\nStop-safely demonstrations (S1):")
    d1 = run_agent(s1_msg, "S1", token_limit=200, run_id="S1_tokenlimit")
    d2 = run_agent(s1_msg, "S1", time_limit_seconds=0, run_id="S1_timeout")
    for d in (d1, d2):
        log_run(DEMOS, d)
        print(f"  {d['run_id']}: status={d['status']} steps_so_far={len(d['steps'])} | {d['final_answer']}")

    print("\nCrash-and-resume demonstration (S1):")
    try:
        run_agent(s1_msg, "S1", run_id="S1_crash", crash_after_tools=2)
    except SimulatedCrash as e:
        print("  crashed:", e)
    res = resume_agent("S1_crash")
    log_run(DEMOS, res)
    print("  tools executed across crash and resume, in order:", res["tools_executed"])
    print(f"  final status={res['status']} refund={refund_amount(res):.2f}")
    assert res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    print("\nledger lines:", sum(1 for _ in open(LEDGER)), "| checkpoints in state/:", len(os.listdir(STATE_DIR)))
