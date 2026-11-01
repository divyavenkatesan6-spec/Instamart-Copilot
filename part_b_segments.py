"""Part B, tasks 1-3: RFM table, K-means (K=2..8), segment profile + names, PCA, outputs/customer_segments.csv."""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
comp = orders[orders["status"] == "completed"]

# ---------- Task 1: RFM ----------
g = comp.groupby("customer_id")
rfm = pd.DataFrame({
    "recency": (SNAPSHOT - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
}).reset_index()
print(f"RFM customers: {len(rfm)} (reference 1175)")
print(f"recency  median={rfm.recency.median():.0f} min={rfm.recency.min()} max={rfm.recency.max()}   (ref 19 / 0 / 180)")
print(f"frequency mean={rfm.frequency.mean():.2f} max={rfm.frequency.max()}   (ref 9.46 / 33)")
print(f"monetary  mean={rfm.monetary.mean():.2f} max={rfm.monetary.max():.2f}   (ref 2552.06 / 14766.55)")

# ---------- Task 2: K-means K=2..8 ----------
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
print("\nK  inertia  silhouette")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(f"{k}  {km.inertia_:7.1f}  {silhouette_score(X, km.labels_):.3f}")

# ---------- Task 3: final model, profile, names, PCA ----------
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), mean_recency=("recency", "mean"),
                                    mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean")).round(2)

# business names assigned by rule from the profile (cluster numbers are only labels, with no ordering)
premium_id = prof["mean_monetary"].idxmax()
names, actions = {premium_id: "Premium Big Spenders"}, {}
rest = prof.drop(index=premium_id)
loyal_id = rest["mean_frequency"].idxmax()
names[loyal_id] = "Loyal Regulars"
rest = rest.drop(index=loyal_id)
lapsing_id = rest["mean_recency"].idxmax()
names[lapsing_id] = "Drifting / At-Risk"
for sid in rest.index:
    if sid != lapsing_id:
        names[sid] = "Occasional Shoppers"
ACTION = {"Premium Big Spenders": "VIP service, early access to electronics, fast support",
          "Loyal Regulars": "Subscription / bundle offers to protect repeat habit",
          "Drifting / At-Risk": "Win-back coupon and reminder before they lapse",
          "Occasional Shoppers": "Light nudges and category cross-sell to raise frequency"}
prof["segment_name"] = [names[i] for i in prof.index]
prof["recommended_action"] = [ACTION[names[i]] for i in prof.index]
print("\nSegment profile (K=4):")
print(prof.to_string())

rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]] \
    .to_csv("outputs/customer_segments.csv", index=False)

pca = PCA(n_components=2).fit(X)
r = pca.explained_variance_ratio_
print(f"\nPCA explained_variance_ratio_: {r[0]:.3f} {r[1]:.3f} total={r.sum():.3f}  (ref 0.634 0.218 0.852)")
print("Premium segment id:", premium_id, "| premium customers:", int(rfm.is_premium.sum()))
print("wrote outputs/customer_segments.csv")
