import streamlit as st
import requests
import joblib

# Page config
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# Title
st.title("🚗 Car Price Prediction App")
st.markdown("**Wondering how much your ride is worth? 🚘 Get an instant estimate of your car’s value anywhere in Morocco — powered by AI!**")

# Load classes
params_encoder = joblib.load("encoders/target_encoding_params.joblib")
brand_classes = list(params_encoder['mappings'].get('brand', {}).keys())
model_classes = list(params_encoder['mappings'].get('model', {}).keys())
origin_classes = list(params_encoder['mappings'].get('origin', {}).keys())

region_list = ['Béni Mellal-Khénifra', 'Casablanca-Settat', 'Dakhla-Oued Ed-Dahab',
               'Drâa-Tafilalet', 'Fès-Meknès', 'Guelmim-Oued Noun', "L'Oriental",
               'Laâyoune-Sakia El Hamra', 'Marrakech-Safi', 'Rabat-Salé-Kénitra',
               'Souss-Massa', 'Tanger-Tétouan-Al Hoceima']

# Input form
st.subheader("📋 Car Details")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", brand_classes)
    model_name = st.selectbox("Model", model_classes)
    origin = st.selectbox("Origin", origin_classes)
    gearbox = st.selectbox("Gearbox", ['automatique', 'manuelle'])
    fuel_type = st.selectbox("Fuel Type", ['diesel', 'essence'])

with col2:
    mileage = st.number_input("Mileage (km)", value=120000, min_value=0)
    fiscal_power = st.number_input("Fiscal Power", value=6, min_value=1)
    condition = st.slider("Condition", min_value=0, max_value=6, value=3, help="0 = Poor, 6 = Excellent")
    year = st.number_input("Year", value=2015, min_value=1990, max_value=2025)
    region = st.selectbox("Region", region_list)

# Prediction button
if st.button("🔍 Predict Price"):
    input_data = {
        "mileage": mileage,
        "brand": brand,
        "model": model_name,
        "origin": origin,
        "fiscal_power": fiscal_power,
        "condition": condition,
        "year": year,
        "gearbox": gearbox,
        "fuel_type": fuel_type,
        "region": region
    }

    try:
        response = requests.post("http://127.0.0.1:8000/price_prediction", json=input_data)
        response_data = response.json()

        if "prediction" in response_data:
            st.success(f"💰 Estimated Price: **{response_data['prediction'][0]:,.0f} MAD**")
        else:
            st.error(f"⚠️ Server Error: {response_data.get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"🚫 Could not connect to the prediction server.\n\n**Details:** {e}")

# Footer
st.markdown("---")
st.markdown("<center><small>🔧 Developed with Streamlit | Powered by FastAPI</small></center>", unsafe_allow_html=True)
