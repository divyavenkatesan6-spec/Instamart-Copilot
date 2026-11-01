"""Part A, task 2: audit the raw data (shape, dtypes, missing values, specific audit figures)."""
import pandas as pd

RAW = "data/raw/"
customers = pd.read_csv(RAW + "customers.csv")
products = pd.read_csv(RAW + "products.csv")
orders = pd.read_csv(RAW + "orders.csv")

for name, df in [("customers", customers), ("products", products), ("orders", orders)]:
    print(f"\n===== {name}.csv  shape={df.shape} =====")
    print("Data types:")
    print(df.dtypes.to_string())
    miss = pd.DataFrame({"missing": df.isna().sum(), "missing_pct": (df.isna().mean() * 100).round(2)})
    print("Missing values per column:")
    print(miss.to_string())

# ---- specific audit figures ----
amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
deliv = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
age = pd.to_numeric(customers["age"], errors="coerce")
gender_nonblank = customers["gender"].dropna()

figs = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "rows with status pending": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    "non-numeric order_amount": int(amount_num.isna().sum()),   # includes "N/A" (read as NaN) and "error"
    "negative delivery_minutes": int((deliv < 0).sum()),
    "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "blank age": int(customers["age"].isna().sum()),
    "age outside 10-100": int(((age < 10) | (age > 100)).sum()),
    "distinct raw city spellings": int(customers["city"].nunique()),
    "distinct non-blank gender spellings": int(gender_nonblank.nunique()),
    "blank gender": int(customers["gender"].isna().sum()),
}
reference = {"raw orders rows": 13084, "exact duplicate rows": 119, "rows with status pending": 656,
             "distinct order_id": 12309, "non-numeric order_amount": 120, "negative delivery_minutes": 106,
             "blank delivery_minutes": 2541, "blank age": 61, "age outside 10-100": 6,
             "distinct raw city spellings": 24, "distinct non-blank gender spellings": 10, "blank gender": 30}

print("\n===== AUDIT FIGURES =====")
print(f"{'figure':40s} {'yours':>8s} {'reference':>10s}  match")
for k, v in figs.items():
    print(f"{k:40s} {v:8d} {reference[k]:10d}  {'OK' if v == reference[k] else 'DIFFERENT'}")
