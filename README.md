🛍️ Smart Retail Analytics & Demand Forecasting

📌 Project Overview

Smart Retail Analytics & Demand Forecasting is a Machine Learning project developed using Python. It analyzes retail sales data, predicts product demand, identifies customer segments, predicts purchase behavior, and detects unusual sales transactions.

🎯 Objectives

- Predict product sales quantity.
- Identify customer segments based on purchasing behavior.
- Predict customer purchase likelihood.
- Detect unusual sales transactions.
- Display insights through an interactive dashboard.

🚀 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- FastAPI
- Matplotlib

🤖 Machine Learning Algorithms

Algorithm| Purpose
Linear Regression| Sales Forecasting
Decision Tree| Purchase Prediction
K-Means| Customer Segmentation
DBSCAN| Anomaly Detection

📊 Project Features

- Sales demand forecasting
- Customer segmentation
- Purchase prediction
- Unusual transaction detection
- Interactive Streamlit dashboard
- FastAPI prediction endpoints

📁 Project Structure

Smart_Retail_Analytics/
│
├── app/
├── data/
├── models/
├── dashboard.py
├── generate_data.py
├── train_models.py
├── requirements.txt
└── README.md

⚙️ Installation & Setup

1. Clone the repository

git clone https://github.com/mansipawar004/Smart_Retail_Analytics.git

2. Open project folder

cd Smart_Retail_Analytics

3. Install required libraries

pip install -r requirements.txt

4. Run Streamlit Dashboard

python -m streamlit run dashboard.py

5. Run FastAPI

python -m uvicorn app.main:app --reload --port 8000

Open API documentation:
http://127.0.0.1:8000/docs

📈 Expected Outcome

The project provides an interactive platform to understand retail sales patterns, forecast demand, analyze customers, and identify unusual transactions.

👩‍💻 Author

Mansi Pawar

## 📸 Dashboard Screenshots

### 1. Demand Forecast
![Demand Forecast](screenshots/demand_forecast.png)

### 2. Purchase Prediction
![Purchase Prediction](screenshots/purchase_prediction.png)

### 3. Customer Segmentation
![Customer Segmentation](screenshots/customer_segment.png)

### 4. Sales Anomaly Detection
![Sales Anomaly](screenshots/sales_anomaly.png)

⚠️ Disclaimer

This project uses synthetic example data for educational and portfolio purposes. Predictions are demonstrations and should not be treated as business-grade forecasts.