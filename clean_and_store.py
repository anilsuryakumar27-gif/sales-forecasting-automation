import pandas as pd
import sqlite3

df = pd.read_csv("data/superstore.csv")
print("Before:", df.shape)

# 1. Remove duplicates
df = df.drop_duplicates()

# 2. Fix data types
df["Order Date"] = pd.to_datetime(df["Order Date"])

# 3. Handle missing values
df["Discount"] = df["Discount"].fillna(0)      # no discount recorded -> 0
df["Region"] = df["Region"].fillna("Unknown")

# 4. Clean column names for SQL
df.columns = [c.replace(" ", "_").replace("-", "_") for c in df.columns]

print("After:", df.shape)
print(df.isnull().sum().sum(), "missing values left")

# 5. Store in SQLite
conn = sqlite3.connect("sales.db")
df.to_sql("sales", conn, if_exists="replace", index=False)

# 6. SQL query: monthly sales
query = """
SELECT strftime('%Y-%m', Order_Date) AS month,
       ROUND(SUM(Sales), 2) AS total_sales,
       ROUND(SUM(Profit), 2) AS total_profit
FROM sales
GROUP BY month
ORDER BY month
"""
monthly = pd.read_sql(query, conn)
print(monthly.head(6))
monthly.to_csv("data/monthly_sales.csv", index=False)
conn.close()