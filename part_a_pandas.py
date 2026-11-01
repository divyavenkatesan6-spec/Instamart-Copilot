"""Part A task 7: pandas cross-check of Q2 and exploratory statistics on completed orders."""
import pandas as pd

o = pd.read_csv("data/clean/orders.csv")
c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")
comp = o[o["status"] == "completed"].merge(c[["customer_id", "city"]], on="customer_id") \
                                    .merge(p[["product_id", "category"]], on="product_id")

# (a) pivot table with margins = Q2 rebuilt
pv = pd.pivot_table(comp, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
print("Pivot grand total:", round(pv.loc["All", "All"], 2), "(SQL Q2 total 2998668.80)")
assert round(pv.loc["All", "All"], 2) == 2998668.80

city = comp.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                                count=("order_amount", "count")).round(2)
print("\nPer city (sum, mean, count):")
print(city.to_string())

# (b) distribution statistics
a = comp["order_amount"]
q1, q2, q3 = a.quantile([.25, .5, .75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean={a.mean():.2f} median={a.median():.2f} skew={a.skew():.2f}")
print(f"Q1={q1:.2f} Q2={q2:.2f} Q3={q3:.2f} IQR={iqr:.2f}")
print(f"lower fence={lo:.2f} outliers below={int((a < lo).sum())} | upper fence={hi:.2f} outliers above={int((a > hi).sum())}")

corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print("\nCorrelation matrix:")
print(corr.round(3).to_string())
cols = list(corr.columns)
pairs = {(a_, b_): corr.loc[a_, b_] for i_, a_ in enumerate(cols) for b_ in cols[i_ + 1:]}
best = max(pairs, key=lambda k: abs(pairs[k]))
print("Largest |correlation| pair:", best, round(pairs[best], 3))
