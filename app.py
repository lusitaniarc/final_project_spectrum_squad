import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Delivery Time Prediction",
    page_icon="🚚",
    layout="centered"
)

# ======================
# LOAD MODEL & FEATURES
# ======================
model = joblib.load("model_final_project.pkl")
final_features = joblib.load("final_features.pkl")

# ======================
# HEADER
# ======================
st.title("🚚 Delivery Time Prediction")
st.caption(
    "Machine learning–based application to estimate food delivery time "
    "based on order details and system conditions."
)

st.markdown("---")

# ======================
# INPUT FORM
# ======================
with st.form("order_form"):
    st.subheader("🧾 Order Information")

    market_id = st.selectbox("Market ID", [1, 2, 3, 4, 5, 6])
    order_protocol = st.selectbox("Order Protocol", [1, 2, 3, 4, 5, 6, 7])

    store_primary_category = st.selectbox(
        "Store Category",
        [
            "afghan", "african", "alcohol", "alcohol-plus-food", "american",
            "argentine", "asian", "barbecue", "belgian", "breakfast",
            "british", "brazilian", "bubble-tea", "burmese", "cafe",
            "cajun", "caribbean", "catering", "cheese", "chinese",
            "chocolate", "comfort-food", "convenience-store",
            "dessert", "dim-sum", "european", "ethiopian", "fast",
            "french", "gastropub", "german", "gluten-free", "greek",
            "hawaiian", "indian", "indonesian", "irish", "italian",
            "japanese", "korean", "kosher", "latin-american", "lebanese",
            "malaysian", "mediterranean", "mexican", "middle-eastern",
            "moroccan", "nepalese", "other", "pakistani", "pasta",
            "persian", "peruvian", "pizza", "russian", "salad",
            "sandwich", "seafood", "singaporean", "smoothie", "soup",
            "southern", "spanish", "steak", "sushi", "tapas",
            "thai", "turkish", "unknown", "vegan", "vegetarian",
            "vietnamese"
        ]
    )

    st.subheader("🛒 Order Details")
    col1, col2 = st.columns(2)
    with col1:
        total_items = st.number_input("Total Items", 1, 50, 4)
        num_distinct_items = st.number_input("Distinct Items", 1, 20, 3)
        subtotal = st.number_input("Subtotal", 0, 50000, 2200)
    with col2:
        min_item_price = st.number_input("Min Item Price", 0, 50000, 500)
        max_item_price = st.number_input("Max Item Price", 0, 50000, 900)

    st.subheader("🚦 System Load")
    col3, col4, col5 = st.columns(3)
    with col3:
        total_onshift_partners = st.number_input("Onshift Partners", 0, 200, 30)
    with col4:
        total_busy_partners = st.number_input("Busy Partners", 0, 200, 25)
    with col5:
        total_outstanding_orders = st.number_input("Outstanding Orders", 0, 300, 60)

    st.subheader("⏰ Order Time")
    col6, col7 = st.columns(2)
    with col6:
        order_date = st.date_input(
            "Order Date",
            value=datetime.date(2015, 2, 6),
            min_value=datetime.date(2015, 1, 1),
            max_value=datetime.date.today()
        )


    with col7:
        order_time = st.time_input(
            "Order Time",
            value=datetime.time(22, 24)
        )

    submitted = st.form_submit_button("🔮 Predict Delivery Time")

# ======================
# PREDICTION PIPELINE
# ======================
if submitted:
    # ----------------------
    # Combine date & time
    # ----------------------
    created_at = datetime.datetime.combine(order_date, order_time)

    # ----------------------
    # Base DataFrame
    # ----------------------
    new_df = pd.DataFrame([{
        "market_id": market_id,
        "order_protocol": order_protocol,
        "store_primary_category": store_primary_category,
        "total_items": total_items,
        "num_distinct_items": num_distinct_items,
        "subtotal": subtotal,
        "min_item_price": min_item_price,
        "max_item_price": max_item_price,
        "total_onshift_partners": total_onshift_partners,
        "total_busy_partners": total_busy_partners,
        "total_outstanding_orders": total_outstanding_orders,
        "created_at": pd.to_datetime(created_at)
    }])

    # ----------------------
    # Feature Engineering
    # ----------------------
    new_df["order_hour"] = new_df["created_at"].dt.hour
    new_df["day_of_week"] = new_df["created_at"].dt.dayofweek
    new_df["is_weekend"] = (new_df["day_of_week"] >= 5).astype(int)

    new_df["load_ratio"] = (
        new_df["total_outstanding_orders"] /
        (new_df["total_onshift_partners"] + 1)
    )

    new_df["busy_partners_ratio"] = (
        new_df["total_busy_partners"] /
        (new_df["total_onshift_partners"] + 1)
    )

    new_df["item_complexity"] = (
        new_df["num_distinct_items"] /
        (new_df["total_items"] + 1)
    )

    new_df["rush_load"] = new_df["load_ratio"] * new_df["is_weekend"]

    # ----------------------
    # Encoding
    # ----------------------
    new_enc = pd.get_dummies(
        new_df,
        columns=["market_id", "order_protocol", "store_primary_category"],
        drop_first=True
    )

    # ----------------------
    # Align Features
    # ----------------------
    new_enc = new_enc.reindex(columns=final_features, fill_value=0)

    # ----------------------
    # Predict
    # ----------------------
    pred = model.predict(new_enc)[0]

    # ======================
    # OUTPUT
    # ======================
    st.markdown("---")

    st.metric(
        label="⏱️ Estimated Delivery Time",
        value=f"{pred:.2f} minutes"
    )

    st.caption(
        f"🕒 Order Date & Time: {created_at.strftime('%Y-%m-%d %H:%M')} | "
        f"Weekend: {'Yes' if new_df['is_weekend'].iloc[0] == 1 else 'No'}"
    )