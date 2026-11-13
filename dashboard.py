import requests
import streamlit as st

st.set_page_config(page_title="Smart Retail Analytics", page_icon="🛍️", layout="wide")
st.title("🛍️ Smart Retail Analytics & Demand Forecasting")
st.caption("Demo dashboard powered by FastAPI and scikit-learn.")

API_URL = st.sidebar.text_input("FastAPI base URL", "http://127.0.0.1:8000")
st.sidebar.info("Start the FastAPI server in a separate terminal before using predictions.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Demand Forecast", "Purchase Prediction", "Customer Segment", "Sales Anomaly"
])

def post_prediction(endpoint, payload):
    try:
        response = requests.post(f"{API_URL.rstrip('/')}/{endpoint}", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the API. Is FastAPI running? Details: {exc}")
        return None

with tab1:
    st.subheader("Predict product demand")
    c1, c2 = st.columns(2)
    price = c1.number_input("Price", min_value=0.01, value=30.0)
    discount = c2.slider("Discount (%)", 0, 100, 10) / 100
    advertising = c1.number_input("Advertising spend", min_value=0.0, value=100.0)
    previous = c2.number_input("Previous sales quantity", min_value=0, value=50)
    if st.button("Forecast sales"):
        result = post_prediction("forecast", {
            "price": price, "discount": discount,
            "advertising_spend": advertising, "previous_sales": int(previous)
        })
        if result:
            st.metric("Predicted sales quantity", result["predicted_sales_quantity"])

with tab2:
    st.subheader("Predict purchase likelihood")
    c1, c2 = st.columns(2)
    p_price = c1.number_input("Product price", min_value=0.01, value=25.0, key="p_price")
    p_discount = c2.slider("Customer discount (%)", 0, 100, 15, key="p_discount") / 100
    p_ad = c1.number_input("Ad spend", min_value=0.0, value=80.0, key="p_ad")
    age = c2.number_input("Customer age", min_value=18, max_value=100, value=30)
    spend = c1.number_input("Customer total spend", min_value=0.0, value=250.0)
    if st.button("Predict purchase"):
        result = post_prediction("purchase", {
            "price": p_price, "discount": p_discount,
            "advertising_spend": p_ad, "customer_age": int(age),
            "customer_spend": spend
        })
        if result:
            st.write(result["purchase_prediction"])
            st.metric("Model probability", f'{result["purchase_probability"]:.1%}')

with tab3:
    st.subheader("Group a customer")
    spend_value = st.number_input("Customer spend", min_value=0.0, value=400.0, key="seg_spend")
    frequency = st.number_input("Purchase frequency", min_value=0, value=12, key="seg_frequency")
    if st.button("Find customer segment"):
        result = post_prediction("segment", {
            "customer_spend": spend_value, "purchase_frequency": int(frequency)
        })
        if result:
            st.metric("Segment ID", result["customer_segment_id"])
            st.caption(result["note"])

with tab4:
    st.subheader("Check for an unusual sales transaction")
    quantity = st.number_input("Sales quantity", min_value=0, value=500)
    a_price = st.number_input("Transaction price", min_value=0.01, value=100.0, key="a_price")
    a_discount = st.slider("Transaction discount (%)", 0, 100, 5, key="a_discount") / 100
    if st.button("Check transaction"):
        result = post_prediction("anomaly", {
            "sales_quantity": quantity, "price": a_price, "discount": a_discount
        })
        if result:
            if result["is_anomaly"]:
                st.warning("This transaction looks unusual and should be reviewed.")
            else:
                st.success("No unusual pattern detected by this demo model.")
            st.json(result)
