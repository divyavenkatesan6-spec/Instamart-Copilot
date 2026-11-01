"""Part D, tasks 1-4: chunk the policy documents, TF-IDF retrieval, grounded answers, evaluation.
Run `python part_d_rag.py` for the report. The refund agent (part_d_agent.py) imports retrieve() from here."""
import glob
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

POLICY_DIR = os.path.join("data", "policies")
CHUNK_WORDS = 40

# ---------------- Task 1: chunking ----------------
def chunk_documents(overlap=10, size=CHUNK_WORDS, folder=POLICY_DIR):
    """Chunks of `size` words; the next chunk starts size-overlap words later; stop at the first chunk that reaches the end."""
    step = size - overlap
    chunks = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        start = 0
        while True:
            chunks.append({"source": os.path.basename(path), "text": " ".join(words[start:start + size])})
            if start + size >= len(words):
                break
            start += step
    return chunks

# ---------------- Task 2: TF-IDF retrieval ----------------
class Retriever:
    def __init__(self, overlap=10):
        self.chunks = chunk_documents(overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(sims[i]), 4), "text": self.chunks[i]["text"]}
                for i in order]

_default = None
def retrieve(query, k=3):
    """Default retriever (40-word chunks, 10-word overlap)."""
    global _default
    if _default is None:
        _default = Retriever(overlap=10)
    return _default.retrieve(query, k)

# ---------------- Task 3: augment and answer ----------------
SYSTEM = ("You are a POLICY ASSISTANT for a grocery delivery company. Answer only from the supplied context. "
          "If the answer is not in the context, say that you could not find the information.")

def answer(question, retriever=None):
    hits = (retriever.retrieve(question, 3) if retriever else retrieve(question, 3))
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, [h["source"] for h in hits]

# ---------------- Task 4: evaluation ----------------
def evaluate(overlap):
    r = Retriever(overlap)
    qa = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    ans_q, none_q = qa[qa["expected_doc"] != "NONE"], qa[qa["expected_doc"] == "NONE"]
    doc_ok = fact_ok = 0
    failures = []
    for _, row in ans_q.iterrows():
        text, sources = answer(row["question"], r)
        d, f = row["expected_doc"] in sources, row["expected_fact"] in text
        doc_ok += d; fact_ok += f
        if not (d and f):
            failures.append((row["question"], d, f, text))
    fallback = sum(answer(q, r)[0].startswith("I could not find") for q in none_q["question"])
    return r, len(ans_q), doc_ok, fact_ok, len(none_q), fallback, failures

if __name__ == "__main__":
    for ov in (10, 0):
        r, n, doc_ok, fact_ok, nn, fb, failures = evaluate(ov)
        counts = pd.Series([c["source"] for c in r.chunks]).value_counts().to_dict()
        print(f"\n===== chunk size 40, overlap {ov} =====")
        print(f"total chunks={len(r.chunks)} {counts}" + ("  (ref 15: refund 4, delivery 3, escalation 3, payment 3, privacy 2)" if ov == 10 else ""))
        print(f"answerable: expected_doc in top-3 = {doc_ok}/{n}; answer contains expected_fact = {fact_ok}/{n}; "
              f"unanswerable fallback = {fb}/{nn}" + ("   (ref 12/12, 12/12, 2/2)" if ov == 10 else ""))
        for q, d, f, text in failures:
            print(f"  FAILS: {q!r} doc_found={d} fact_found={f}\n         answer: {text}")
    print("\nTop 3 for 'refund policy late delivery':", [h["source"] for h in retrieve("refund policy late delivery")], "(ref: all refund_policy.txt)")
