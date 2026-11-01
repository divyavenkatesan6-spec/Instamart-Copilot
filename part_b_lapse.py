"""Part B, tasks 4-8: lapse prediction (no leakage), tuning, class-balance experiment, interpretation, persistence."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.makedirs("outputs", exist_ok=True)
CUTOFF = pd.Timestamp("2025-09-01")
SEP_END = pd.Timestamp("2025-10-01")

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
products = pd.read_csv("data/clean/products.csv")
orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

# ---------- Task 4: cohort, label, features (ONLY orders before the cutoff) ----------
before = orders[orders["order_ts"] < CUTOFF]                      # nothing dated on/after 2025-09-01 is used for features
comp_b = before[before["status"] == "completed"]
cohort_ids = comp_b["customer_id"].unique()

g = comp_b.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount": g["discount_pct"].mean(),
    "avg_delivery": g["delivery_minutes"].mean(),
    "fav_category": g["category"].agg(lambda s: s.value_counts().idxmax()),
})
all_b = before.groupby("customer_id")
feat["cancel_rate"] = (all_b["status"].apply(lambda s: (s == "cancelled").sum()) / all_b["status"].count()).reindex(feat.index)
feat = feat.reset_index().merge(customers[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id")
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
feat = feat.drop(columns="signup_date")

sept = orders[(orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < SEP_END) & (orders["status"] == "completed")]
active_sept = set(sept["customer_id"])
feat["lapsed"] = (~feat["customer_id"].isin(active_sept)).astype(int)     # label only; never a feature

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X, y = feat[NUM + CAT], feat["lapsed"]
print(f"cohort={len(feat)} (ref 1156) lapsed={int(y.sum())} ({y.mean()*100:.2f}%) (ref 443, 38.32%)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
ids_test = feat.loc[X_test.index, "customer_id"]
print(f"train={len(X_train)} test={len(X_test)} (ref 924 / 232)")

# ---------- Task 5: preprocessing without leakage ----------
def make_pre():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
assert X.isna().sum().sum() == 0, "feature columns contain missing values"
print("No missing values in any feature column: confirmed")
pre_check = make_pre().fit(X_train)                                  # fitted on TRAIN rows only
learned = pre_check.named_transformers_["num"].mean_[NUM.index("frequency")]
print(f"scaler mean of frequency (train only) = {learned:.4f} | full-cohort mean = {feat['frequency'].mean():.4f}")

# ---------- Task 6: baseline, tuning, evaluation ----------
def metrics(model, Xt, yt):
    p = model.predict(Xt)
    return dict(precision=precision_score(yt, p, zero_division=0), recall=recall_score(yt, p, zero_division=0),
                f1=f1_score(yt, p, zero_division=0), cm=confusion_matrix(yt, p))

base = np.zeros(len(y_test), dtype=int)
print(f"\nBaseline (always 'not lapsed'): accuracy={accuracy_score(y_test, base)*100:.1f}% "
      f"F1={f1_score(y_test, base, zero_division=0):.2f} (ref 61.6% / 0)")

lr_pipe = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000))])
lr_gs = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_train, y_train)
rf_pipe = Pipeline([("pre", make_pre()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_gs = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                     cv=5, scoring="f1").fit(X_train, y_train)
print(f"\nLogReg best params={lr_gs.best_params_} CV F1={lr_gs.best_score_:.3f}")
print(f"RandForest best params={rf_gs.best_params_} CV F1={rf_gs.best_score_:.3f}")

res = {"LogisticRegression": metrics(lr_gs.best_estimator_, X_test, y_test),
       "RandomForest": metrics(rf_gs.best_estimator_, X_test, y_test)}
print("\nTEST set, lapsed class (tuning is finished):")
for name, m in res.items():
    print(f"{name}: precision={m['precision']:.3f} recall={m['recall']:.3f} F1={m['f1']:.3f}")
    print("  confusion matrix [[TN FP],[FN TP]]:", m["cm"].tolist())

chosen_name = max(res, key=lambda k: res[k]["f1"])
best = {"LogisticRegression": lr_gs, "RandomForest": rf_gs}[chosen_name].best_estimator_
print(f"\nSelected model: {chosen_name} (test F1={res[chosen_name]['f1']:.3f}; must be >= 0.55)")
assert res[chosen_name]["f1"] >= 0.55

# ---------- Task 7: class-balance experiment (oversample TRAIN only) ----------
lapsed_idx = y_train[y_train == 1].index
n_add = int((y_train == 0).sum() - (y_train == 1).sum())
extra = np.random.RandomState(42).choice(lapsed_idx, size=n_add, replace=True)
bal_idx = list(y_train.index) + list(extra)
X_bal, y_bal = X.loc[bal_idx], y.loc[bal_idx]
print(f"\nBalanced train class counts: {y_bal.value_counts().to_dict()} (ref 570 each)")
lr_best = clone(lr_gs.best_estimator_).fit(X_bal, y_bal)
m0, m1 = res["LogisticRegression"], metrics(lr_best, X_test, y_test)
print(f"LogReg un-balanced: precision={m0['precision']:.3f} recall={m0['recall']:.3f} F1={m0['f1']:.3f}")
print(f"LogReg balanced   : precision={m1['precision']:.3f} recall={m1['recall']:.3f} F1={m1['f1']:.3f}")

# ---------- Task 8: interpretation, persistence, retention list ----------
rf_best = rf_gs.best_estimator_
names = rf_best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop 5 random-forest features:")
print(imp.head(5).round(4).to_string())
lr_b = lr_gs.best_estimator_
coef = pd.Series(lr_b.named_steps["clf"].coef_[0], index=lr_b.named_steps["pre"].get_feature_names_out())
print("\nLogistic regression coefficient signs (positive = more likely to lapse):")
for f in ["num__recency", "num__frequency", "num__monetary"]:
    print(f"  {f}: {coef[f]:+.3f}")

joblib.dump(best, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
assert (reloaded.predict(X_test) == best.predict(X_test)).all()
print("\nReloaded model reproduces the original test predictions exactly: True")

seg = pd.read_csv("outputs/customer_segments.csv")
prem = set(seg.loc[seg["is_premium"], "customer_id"])
test_df = pd.DataFrame({"customer_id": ids_test.values, "lapse_probability": best.predict_proba(X_test)[:, 1],
                        "actually_lapsed": y_test.values})
ret = test_df[test_df["customer_id"].isin(prem)].sort_values("lapse_probability", ascending=False).head(10)
ret["lapse_probability"] = ret["lapse_probability"].round(4)
ret.to_csv("outputs/retention_list.csv", index=False)
print(f"Retention list: {len(ret)} premium test customers; truly lapsed: {int(ret['actually_lapsed'].sum())}")
print("wrote outputs/lapse_model.joblib and outputs/retention_list.csv")
