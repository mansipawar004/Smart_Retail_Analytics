from pathlib import Path
import joblib
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "retail_sales.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError("Dataset missing. Run: python generate_data.py")

df = pd.read_csv(DATA)

# 1) Sales forecasting: predict quantity from sales/marketing variables.
forecast_features = ["price", "discount", "advertising_spend", "previous_sales"]
X = df[forecast_features]
y = df["sales_quantity"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
forecast_model = LinearRegression()
forecast_model.fit(X_train, y_train)
forecast_pred = forecast_model.predict(X_test)
print(f"Forecast MAE: {mean_absolute_error(y_test, forecast_pred):.2f}")
joblib.dump({
    "model": forecast_model,
    "features": forecast_features
}, MODEL_DIR / "forecast_model.joblib")

# 2) Purchase prediction: binary classifier.
purchase_features = [
    "price", "discount", "advertising_spend", "customer_age", "customer_spend"
]
X = df[purchase_features]
y = df["purchase"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
purchase_model = DecisionTreeClassifier(
    max_depth=5, random_state=42, class_weight="balanced"
)
purchase_model.fit(X_train, y_train)
print(f"Purchase model accuracy: {accuracy_score(y_test, purchase_model.predict(X_test)):.3f}")
joblib.dump({
    "model": purchase_model,
    "features": purchase_features
}, MODEL_DIR / "purchase_model.joblib")

# 3) Customer segmentation: K-Means on spending and purchase frequency.
segment_features = ["customer_spend", "purchase_frequency"]
segment_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=3, random_state=42, n_init=10))
])
segment_pipeline.fit(df[segment_features])
joblib.dump({
    "model": segment_pipeline,
    "features": segment_features
}, MODEL_DIR / "segment_model.joblib")

# 4) Unusual transaction detection: DBSCAN labels outliers as -1.
anomaly_features = ["sales_quantity", "price", "discount"]
anomaly_scaler = StandardScaler()
scaled = anomaly_scaler.fit_transform(df[anomaly_features])
anomaly_model = DBSCAN(eps=0.65, min_samples=8)
anomaly_model.fit(scaled)
joblib.dump({
    "scaler": anomaly_scaler,
    "model": anomaly_model,
    "features": anomaly_features,
    "training_data": scaled
}, MODEL_DIR / "anomaly_model.joblib")

print("Saved all four models to the models/ folder.")
