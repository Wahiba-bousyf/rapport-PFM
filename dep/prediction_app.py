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

# Style CSS personnalisé avancé
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

body, .stApp {
    font-family: 'Poppins', sans-serif;
    background-color: #f9f9f9;
    color: #333;
}

.form-card, .result-card {
    background: white;
    border-radius: 12px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    padding: 25px 30px;
    margin-bottom: 30px;
    transition: box-shadow 0.3s ease;
}

.form-card:hover, .result-card:hover {
    box-shadow: 0 12px 30px rgba(0,0,0,0.15);
}

.stButton>button {
    background: linear-gradient(90deg, #007BFF 0%, #0056b3 100%);
    color: white;
    font-weight: 600;
    border-radius: 8px;
    padding: 12px 25px;
    border: none;
    box-shadow: 0 4px 12px rgba(0,123,255,0.4);
    cursor: pointer;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #0056b3 0%, #003d80 100%);
    box-shadow: 0 6px 18px rgba(0,86,179,0.6);
}

.stSelectbox > div, .stNumberInput > div {
    border-radius: 8px !important;
    border: 1.5px solid #ddd !important;
    padding: 8px 12px !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    transition: border-color 0.3s ease;
}

.stSelectbox > div:focus-within, .stNumberInput > div:focus-within {
    border-color: #007BFF !important;
    box-shadow: 0 0 8px rgba(0,123,255,0.4) !important;
}

li {
    padding: 10px 16px !important;
    font-size: 0.95rem !important;
    transition: background-color 0.3s ease, color 0.3s ease;
}

li:hover {
    background-color: #007BFF !important;
    color: white !important;
}

.header h1 {
    font-weight: 700;
    color: #007BFF;
    margin-bottom: 0.2rem;
}

.header p {
    font-size: 1.1rem;
    color: #555;
    margin-top: 0;
    margin-bottom: 1.5rem;
}

.result-price {
    font-size: 2.5rem;
    font-weight: 700;
    color: #28a745;
    margin-top: 0.3rem;
    margin-bottom: 0.3rem;
}

footer {
    text-align: center;
    font-size: 0.9rem;
    padding: 15px 0;
    color: #666;
    border-top: 1px solid #eee;
    margin-top: 50px;
}

* {
    transition: all 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

# Chargement des classes - avec vérification des données
try:
    params_encoder = joblib.load("encoders/target_encoding_params.joblib")
    brand_classes = list(params_encoder['mappings'].get('brand', {}).keys())
    model_classes = list(params_encoder['mappings'].get('model', {}).keys())
    origin_classes = list(params_encoder['mappings'].get('origin', {}).keys())
    
    # Vérification et nettoyage des données
    brand_classes = [str(x) for x in brand_classes if x is not None]
    model_classes = [str(x) for x in model_classes if x is not None]
    origin_classes = [str(x) for x in origin_classes if x is not None]
    
    encoder_region = joblib.load('encodage_apres_equilibre/label_encoder_final_region.joblib')
    region_classes = list(encoder_region.classes_)
    region_display = ['Autre' if x == '-1' else str(x) for x in region_classes if x is not None]
    
    condition_options = ['neuf', 'excellent', 'très bon', 'bon']
    condition_mapping = joblib.load('encodage_apres_equilibre/condition_mapping.joblib')
    
except Exception as e:
    st.error(f"Erreur lors du chargement des données: {str(e)}")
    st.stop()

# En-tête
st.markdown("""
<div class="header">
    <h1>🚗 Estimation du Prix des Véhicules</h1>
    <p>Obtenez une estimation instantanée de la valeur de votre véhicule au Maroc grâce à notre technologie d'IA avancée.</p>
</div>
""", unsafe_allow_html=True)

# Section d'information avant le formulaire
with st.expander("ℹ️ Comment obtenir la meilleure estimation", expanded=False):
    st.markdown("""
    **Pour une estimation précise :**
    - Renseignez tous les champs avec exactitude
    - Le kilométrage doit correspondre à celui visible sur le compteur
    - Sélectionnez l'état qui correspond vraiment à votre véhicule
    
    **Échelle d'état du véhicule :**
    - **Neuf** : Véhicule jamais utilisé, encore sous garantie constructeur
    - **Excellent** : Comme neuf, très peu de traces d'usage, intérieur impeccable
    - **Très bon** : Quelques légères traces d'usage mais en excellent état général
    - **Bon** : Usure normale pour l'âge du véhicule, peut nécessiter quelques retouches
    
    **Note :** Une estimation en personne par un expert peut varier de ±10% par rapport à cette estimation en ligne.
    """)


# Formulaire dans une carte
with st.container():
    #st.markdown('<div class="form-card">', unsafe_allow_html=True)
    
    st.subheader("📋 Détails du Véhicule")
    col1, col2 = st.columns(2)

    with col1:
        try:
            brand = st.selectbox(
                "Marque", 
                options=brand_classes,
                index=0 if brand_classes else None,
                help="Sélectionnez la marque de votre véhicule"
            )
            
            model_name = st.selectbox(
                "Modèle", 
                options=model_classes,
                index=0 if model_classes else None,
                help="Sélectionnez le modèle de votre véhicule"
            )
            
            origin = st.selectbox(
                "Origine", 
                options=origin_classes,
                index=0 if origin_classes else None,
                help="Sélectionnez l'origine du véhicule"
            )
            
            gearbox = st.selectbox(
                "Boîte de vitesses", 
                options=['automatique', 'manuelle'], 
                index=0,
                help="Type de transmission"
            )
            
            fuel_type = st.selectbox(
                "Type de carburant", 
                options=['diesel', 'essence', 'Autre'], 
                index=0,
                help="Sélectionnez le type de carburant"
            )

        except Exception as e:
            st.error(f"Erreur dans la colonne 1: {str(e)}")

    with col2:
        try:
            mileage = st.number_input(
                "Kilométrage (km)", 
                value=120000, 
                min_value=0, 
                step=1000, 
                help="Kilométrage actuel du véhicule"
            )
            
            fiscal_power = st.number_input(
                "Puissance fiscale", 
                value=6, 
                min_value=1, 
                help="Puissance fiscale du véhicule"
            )
            
            condition = st.selectbox(
                "État du véhicule",
                options=condition_options,
                index=0,
                help="Sélectionnez l'état général du véhicule"
            )
            
            year = st.number_input(
                "Année", 
                value=2015, 
                min_value=1990, 
                max_value=2025, 
                step=1, 
                help="Année de fabrication"
            )
            
            region = st.selectbox(
                "Région", 
                options=region_display,
                index=0 if region_display else None,
                help="Région où se trouve le véhicule"
            )
            
        except Exception as e:
            st.error(f"Erreur dans la colonne 2: {str(e)}")
    
    region_to_send = '-1' if region == 'Autre' else region
    
    st.markdown('</div>', unsafe_allow_html=True)

# Bouton de prédiction
if st.button("🔍 Estimer le Prix"):
    try:
        input_data = {
            "mileage": mileage,
            "brand": brand,
            "model": model_name,
            "origin": origin,
            "fiscal_power": fiscal_power,
            "condition": condition_mapping.get(condition, 0),
            "year": year,
            "gearbox": gearbox,
            "fuel_type": fuel_type,
            "region": region_to_send
        }

        with st.spinner('Calcul de l\'estimation en cours...'):
            response = requests.post("http://127.0.0.1:8000/price_prediction", json=input_data)
            response.raise_for_status()
            response_data = response.json()

        if "prediction" in response_data:
            st.markdown(f"""
            <div class="result-card">
                <h3>Estimation du Prix</h3>
                <div class="result-price">{response_data['prediction'][0]:,.0f} MAD</div>
                <p>Cette estimation est calculée en temps réel par notre algorithme d'IA</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"⚠️ Erreur du serveur: {response_data.get('error', 'Réponse inattendue du serveur')}")
            
    except requests.exceptions.RequestException as e:
        st.error(f"🚫 Erreur de connexion: {str(e)}")
    except Exception as e:
        st.error(f"🚫 Erreur inattendue: {str(e)}")

# Pied de page
st.markdown("""
<footer>
    <p>🔧 Développé avec Streamlit | Propulsé par FastAPI</p>
</footer>
""", unsafe_allow_html=True)