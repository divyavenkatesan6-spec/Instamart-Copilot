"""Part A task 6: run the seven queries in sql/queries.sql and print the stated results."""
import re
import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql", encoding="utf-8").read()
blocks = [b.strip() for b in re.split(r"(?m)^-- Q\d+:.*$", text) if b.strip()]
q = [pd.read_sql_query(b, con) for b in blocks]
assert len(q) == 7, "expected 7 queries"

print("Q1 customers:", len(q[0]), "(reference 30)")
print("Q2 rows:", len(q[1]), "| total:", round(q[1].revenue.sum(), 2), "| largest:",
      q[1].iloc[0].city, "/", q[1].iloc[0].category, q[1].iloc[0].revenue, "(reference 30, 2998668.80, Bengaluru/Electronics 233909.45)")
print("Q3 products:", len(q[2]), "(reference 11)")
print("Q4 customers with no completed order:", len(q[3]), "(reference 25)")
print("Q5 customers above average:", len(q[4]), "| average:", q[4].avg_total.iloc[0], "(reference 550, 2552.06)")
print("Q6 top customer per city:")
print(q[5][q[5].row_num == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows where RANK <> ROW_NUMBER:", int((q[5].rnk != q[5].row_num).sum()), "(reference 8)")
print("Q7 daily rows:", len(q[6]), "| last cumulative:", q[6].cumulative_revenue.iloc[-1], "(reference 183, 2998668.80)")
con.close()
