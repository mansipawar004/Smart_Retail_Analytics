from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 1200
OUT = Path(__file__).parent / "data" / "retail_sales.csv"
OUT.parent.mkdir(exist_ok=True)

price = RNG.uniform(5, 150, N).round(2)
discount = RNG.uniform(0, 0.40, N).round(2)
advertising_spend = RNG.uniform(0, 500, N).round(2)
previous_sales = RNG.integers(5, 180, N)
customer_age = RNG.integers(18, 75, N)
customer_spend = RNG.gamma(shape=3.0, scale=80.0, size=N).round(2)
purchase_frequency = RNG.integers(1, 30, N)

noise = RNG.normal(0, 12, N)
sales_quantity = (
    65 + previous_sales * 0.48 - price * 0.20
    + discount * 65 + advertising_spend * 0.035 + noise
)
sales_quantity = np.clip(np.rint(sales_quantity), 0, None).astype(int)

# Add a few deliberately unusual transactions for anomaly-detection practice.
anomaly_idx = RNG.choice(N, size=18, replace=False)
sales_quantity[anomaly_idx[:9]] *= 4
price[anomaly_idx[9:]] *= 3

purchase_probability = (
    0.15 + discount * 0.8 + advertising_spend / 1200
    + customer_spend / 1500 - price / 500
)
purchase_probability = np.clip(purchase_probability, 0.05, 0.95)
purchase = RNG.binomial(1, purchase_probability)

df = pd.DataFrame({
    "customer_id": [f"C{i:04d}" for i in RNG.integers(1, 450, N)],
    "product_id": [f"P{i:03d}" for i in RNG.integers(1, 80, N)],
    "price": price,
    "discount": discount,
    "advertising_spend": advertising_spend,
    "previous_sales": previous_sales,
    "sales_quantity": sales_quantity,
    "customer_age": customer_age,
    "customer_spend": customer_spend,
    "purchase_frequency": purchase_frequency,
    "purchase": purchase,
})
df.to_csv(OUT, index=False)
print(f"Created {OUT} with {len(df)} sample rows.")
