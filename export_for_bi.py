import sqlite3
import pandas as pd

conn = sqlite3.connect("sales.db")
pd.read_sql("SELECT * FROM sales", conn).to_csv("data/sales_clean.csv", index=False)
print("Exported")