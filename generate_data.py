import numpy as np
import pandas as pd

np.random.seed(42)
n = 6000

dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(
    np.random.randint(0, 1095, n), unit="D")

cats = {
    "Technology": (["Phones", "Accessories", "Machines"], 300, 0.25),
    "Furniture": (["Chairs", "Tables", "Bookcases"], 250, 0.10),
    "Office Supplies": (["Paper", "Binders", "Storage"], 40, 0.30),
}
cat = np.random.choice(list(cats), n, p=[0.3, 0.3, 0.4])
sub = [np.random.choice(cats[c][0]) for c in cat]
price = np.array([cats[c][1] for c in cat]) * np.random.uniform(0.6, 1.6, n)
margin = np.array([cats[c][2] for c in cat])

qty = np.random.randint(1, 9, n)
disc = np.random.choice([0, 0.1, 0.2, 0.3, 0.4], n, p=[0.4, 0.2, 0.2, 0.1, 0.1])
season = np.where(dates.month.isin([11, 12]), 1.4,
         np.where(dates.month.isin([6, 7]), 0.9, 1.0))
trend = 1 + (dates - dates.min()).days / 1095 * 0.4

sales = price * qty * (1 - disc) * season * trend
profit = sales * (margin - disc * 0.6) + np.random.normal(0, 5, n)

df = pd.DataFrame({
    "Order ID": [f"ORD-{i:05d}" for i in range(n)],
    "Order Date": dates.strftime("%Y-%m-%d"),
    "Region": np.random.choice(["East", "West", "Central", "South"], n),
    "Segment": np.random.choice(["Consumer", "Corporate", "Home Office"], n),
    "Category": cat,
    "Sub-Category": sub,
    "Quantity": qty,
    "Discount": disc,
    "Sales": sales.round(2),
    "Profit": profit.round(2),
})

# add realistic mess for the cleaning step
df.loc[np.random.choice(n, 60, replace=False), "Discount"] = np.nan
df.loc[np.random.choice(n, 40, replace=False), "Region"] = np.nan
df = pd.concat([df, df.sample(25, random_state=1)], ignore_index=True)

df.to_csv("data/superstore.csv", index=False)
print("Saved:", df.shape)