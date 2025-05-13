import streamlit as st
import requests
import joblib
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Style CSS professionnel
st.markdown("""
<style>
    :root {
        --primary: #2563eb;
        --primary-light: #3b82f6;
        --primary-dark: #1d4ed8;
        --secondary: #f8fafc;
        --accent: #10b981;
        --text: #1e293b;
        --text-light: #64748b;
        --background: #ffffff;
        --card: #f9fafb;
        --border: #e2e8f0;
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
    }

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    body, .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        color: var(--text);
    }

    /* Header */
    .header {
        text-align: center;
        margin-bottom: 3rem;
    }

    .header h1 {
        color: var(--primary);
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
    }

    .header p {
        color: var(--text-light);
        font-size: 1.1rem;
        max-width: 700px;
        margin: 0 auto;
    }

    /* Form Card */
    .form-card {
        background: var(--background);
        border-radius: var(--radius-lg);
        padding: 2.5rem;
        box-shadow: var(--shadow-lg);
        border: 1px solid var(--border);
        margin-bottom: 2rem;
    }

    /* Inputs */
    .stSelectbox, .stNumberInput {
        margin-bottom: 1.5rem;
    }

    /* Labels */
    .st-ae label {
        font-weight: 600 !important;
        color: var(--text) !important;
        margin-bottom: 0.5rem !important;
        font-size: 0.95rem !important;
    }

    /* Select Box */
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border) !important;
        padding: 10px 14px !important;
        font-size: 0.95rem !important;
        background: var(--background) !important;
        color: var(--text) !important;
    }

    /* Input Fields */
    .stNumberInput input, .stTextInput input {
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border) !important;
        padding: 10px 14px !important;
        font-size: 0.95rem !important;
        background: var(--background) !important;
    }

    /* Focus States */
    .stSelectbox div[data-baseweb="select"] > div:focus-within,
    .stNumberInput input:focus,
    .stTextInput input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }

    /* Dropdown Menu */
    div[data-baseweb="menu"] {
        border-radius: var(--radius-sm) !important;
        margin-top: 8px !important;
        box-shadow: var(--shadow-md) !important;
        border: 1px solid var(--border) !important;
    }

    div[data-baseweb="menu"] li {
        padding: 10px 16px !important;
        font-size: 0.95rem !important;
    }

    div[data-baseweb="menu"] li:hover {
        background-color: var(--secondary) !important;
        color: var(--primary) !important;
    }

    /* Button */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }

    .stButton button:hover {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%) !important;
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }

    /* Result Card */
    .result-card {
        background: var(--card);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        text-align: center;
        margin-top: 1.5rem;
        border-left: 4px solid var(--primary);
        box-shadow: var(--shadow-sm);
    }

    .result-card h3 {
        color: var(--primary);
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .result-price {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text);
        margin: 0.5rem 0;
    }

    /* Columns */
    .st-cq, .st-dk, .st-ez {
        padding: 0 1rem;
    }

    /* Footer */
    footer {
        text-align: center;
        margin-top: 3rem;
        color: var(--text-light);
        font-size: 0.9rem;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .header h1 {
            font-size: 2rem;
        }
        
        .header p {
            font-size: 1rem;
        }
        
        .form-card {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Chargement des classes
params_encoder = joblib.load("encoders/target_encoding_params.joblib")
brand_classes = list(params_encoder['mappings'].get('brand', {}).keys())
model_classes = list(params_encoder['mappings'].get('model', {}).keys())
origin_classes = list(params_encoder['mappings'].get('origin', {}).keys())

encoder_region = joblib.load('encodage_apres_equilibre/label_encoder_final_region.joblib')
region_classes = list(encoder_region.classes_)
region_display = ['Autre' if x == '-1' else x for x in region_classes]

condition_options = ['neuf', 'excellent', 'très bon', 'bon']
condition_mapping = joblib.load('encodage_apres_equilibre/condition_mapping.joblib')

# En-tête
st.markdown("""
<div class="header">
    <h1>🚗 Car Price Prediction</h1>
    <p>Get an instant estimate of your car's value in Morocco with our AI-powered valuation tool.</p>
</div>
""", unsafe_allow_html=True)

# Formulaire
with st.container():
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    
    st.subheader("📋 Vehicle Details")
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", brand_classes, help="Select the vehicle brand")
        model_name = st.selectbox("Model", model_classes, help="Select the vehicle model")
        origin = st.selectbox("Origin", origin_classes, help="Select the vehicle origin")
        gearbox = st.selectbox("Gearbox", ['automatique', 'manuelle'], help="Select transmission type")
        fuel_type = st.selectbox("Fuel Type", ['diesel', 'essence', 'Autre'], help="Select fuel type")

    with col2:
        mileage = st.number_input("Mileage (km)", value=120000, min_value=0, step=1000, help="Current vehicle mileage")
        fiscal_power = st.number_input("Fiscal Power", value=6, min_value=1, help="Vehicle fiscal power")
        condition = st.selectbox(
            "Vehicle Condition",
            options=condition_options,
            index=0,
            help="Select the vehicle condition"
        )
        year = st.number_input("Year", value=2015, min_value=1990, max_value=2025, step=1, help="Manufacturing year")
        region = st.selectbox("Region", options=region_display, help="Vehicle location region")
    
    region_to_send = '-1' if region == 'Autre' else region
    
    st.markdown('</div>', unsafe_allow_html=True)

# Bouton de prédiction
if st.button("🔍 Predict Price"):
    input_data = {
        "mileage": mileage,
        "brand": brand,
        "model": model_name,
        "origin": origin,
        "fiscal_power": fiscal_power,
        "condition": condition_mapping[condition],
        "year": year,
        "gearbox": gearbox,
        "fuel_type": fuel_type,
        "region": region_to_send
    }

    try:
        with st.spinner('Calculating estimate...'):
            response = requests.post("http://127.0.0.1:8000/price_prediction", json=input_data)
            response_data = response.json()

        if "prediction" in response_data:
            st.markdown(f"""
            <div class="result-card">
                <h3>Estimated Price</h3>
                <div class="result-price">{response_data['prediction'][0]:,.0f} MAD</div>
                <p>This estimate is calculated in real-time by our AI algorithm</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"⚠️ Server Error: {response_data.get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"🚫 Could not connect to prediction server.\n\n**Details:** {e}")

# Pied de page
st.markdown("""
<footer>
    <p>🔧 Developed with Streamlit | Powered by FastAPI</p>
</footer>
""", unsafe_allow_html=True)