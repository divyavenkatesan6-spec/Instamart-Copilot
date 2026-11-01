"""Part A, tasks 3-5: clean customers and orders, write data/clean/*.csv and data/instamart.db."""
import os
import sqlite3
import numpy as np
import pandas as pd

RAW, CLEAN = "data/raw/", "data/clean/"
os.makedirs(CLEAN, exist_ok=True)

customers = pd.read_csv(RAW + "customers.csv")
products = pd.read_csv(RAW + "products.csv")
orders = pd.read_csv(RAW + "orders.csv")

# =============== CUSTOMERS ===============
# (a) city: strip spaces + title case
before = customers["city"].nunique()
customers["city"] = customers["city"].str.strip().str.title()
print(f"[customers] city spellings: {before} -> {customers['city'].nunique()}")

# (b) gender mapping (blank -> unknown)
def map_gender(g):
    if pd.isna(g) or str(g).strip() == "":
        return "unknown"
    g = str(g).strip().lower()
    if g in ("m", "male"):
        return "male"
    if g in ("f", "female"):
        return "female"
    return "unknown"
customers["gender"] = customers["gender"].apply(map_gender)

# (c) ages outside 10-100 -> missing, then fill with city median (groupby + transform)
bad_age = (customers["age"] < 10) | (customers["age"] > 100)
print(f"[customers] ages outside 10-100 set to missing: {int(bad_age.sum())}")
customers.loc[bad_age, "age"] = np.nan
n_missing = int(customers["age"].isna().sum())
customers["age"] = customers["age"].fillna(customers.groupby("city")["age"].transform("median"))
print(f"[customers] missing ages filled with city median: {n_missing}")

# (d) signup_date -> datetime
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# =============== ORDERS ===============
n0 = len(orders)
# 1. order_ts -> datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# 2. remove exact duplicate rows
orders = orders.drop_duplicates()
print(f"[orders] exact duplicates removed: {n0 - len(orders)}")

# 3. keep latest record per order_id
n1 = len(orders)
orders = orders.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
print(f"[orders] older pending/partial records removed: {n1 - len(orders)}")

# 4. order_amount -> numeric, fill missing from product price
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
orders = orders.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left")
miss_amt = orders["order_amount"].isna()
orders.loc[miss_amt, "order_amount"] = (
    orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)[miss_amt]
print(f"[orders] unparseable amounts filled from product price: {int(miss_amt.sum())}")

# 5. delivery_minutes: negatives -> missing; fill completed by category median (groupby + transform)
orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
neg = orders["delivery_minutes"] < 0
orders.loc[neg, "delivery_minutes"] = np.nan
print(f"[orders] negative delivery_minutes set to missing: {int(neg.sum())}")
completed = orders["status"] == "completed"
cat_median = orders.loc[completed].groupby("category")["delivery_minutes"].transform("median")
need = completed & orders["delivery_minutes"].isna()
orders.loc[need, "delivery_minutes"] = cat_median.loc[need[need].index]
print(f"[orders] completed orders given category-median delivery time: {int(need.sum())}")
print("[orders] cancelled orders left with missing delivery_minutes (never delivered)")

orders = orders.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")   # text form for CSV and SQLite

# =============== STORE ===============
customers_out = customers.copy()
customers_out["signup_date"] = customers_out["signup_date"].dt.strftime("%Y-%m-%d")
customers_out.to_csv(CLEAN + "customers.csv", index=False)
products.to_csv(CLEAN + "products.csv", index=False)
orders.to_csv(CLEAN + "orders.csv", index=False)

if os.path.exists("data/instamart.db"):
    os.remove("data/instamart.db")
con = sqlite3.connect("data/instamart.db")
customers_out.to_sql("customers", con, index=False)
products.to_sql("products", con, index=False)
orders.to_sql("orders", con, index=False)
con.close()
print(f"[store] wrote data/clean/*.csv and data/instamart.db  (orders={len(orders)}, customers={len(customers_out)})")
