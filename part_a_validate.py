"""Part A task 5: validation script. Prints exactly ALL CHECKPOINTS PASSED when every assertion holds."""
import sqlite3
import pandas as pd

c = pd.read_csv("data/clean/customers.csv")
o = pd.read_csv("data/clean/orders.csv")
comp, canc = o[o["status"] == "completed"], o[o["status"] == "cancelled"]

assert len(o) == 12309 and o["order_id"].is_unique, "orders must have 12,309 unique rows"
assert (o["status"] == "completed").sum() == 11110, "completed count"
assert (o["status"] == "cancelled").sum() == 1199, "cancelled count"
assert not (o["status"] == "pending").any(), "no pending rows"
assert o["order_amount"].isna().sum() == 0, "no missing order_amount"
assert comp["delivery_minutes"].isna().sum() == 0, "no missing delivery among completed"
assert canc["delivery_minutes"].isna().sum() == 1199, "1,199 missing delivery among cancelled"
assert round(comp["order_amount"].sum(), 2) == 2998668.80, "completed revenue"
assert c["city"].value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                                              "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}, "city counts"
assert c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}, "gender counts"
assert c["age"].isna().sum() == 0 and c["age"].between(18, 60).all(), "ages"

con = sqlite3.connect("data/instamart.db")
tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
assert {"customers", "products", "orders"} <= tables, "db tables"
assert con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 12309, "db orders rows"
con.close()
print("ALL CHECKPOINTS PASSED")
