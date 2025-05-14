

from fastapi import FastAPI, Request
import joblib
import numpy as np
import json
from typing import Dict, Any

app = FastAPI()

# Chargement des modèles et encodeurs
model = joblib.load("models/lgbm_model.pkl")
scaler = joblib.load("scaler/scaler.pkl")
fuel_encoder = joblib.load("encodage_apres_equilibre/label_encoder_final_fuel_type.joblib")
region_encoder = joblib.load("encodage_apres_equilibre/label_encoder_final_region.joblib")
gearbox_encoder = joblib.load("encoders/label_encoder_gearbox.joblib")
condition_mapping = joblib.load('encodage_apres_equilibre/condition_mapping.joblib')
params_encoder = joblib.load("encoders/target_encoding_params.joblib")

def target_encode_smooth_deploy(cat_col: str, value: str, global_mean: float, mappings: Dict[str, Any]) -> float:
    return mappings.get(cat_col, {}).get(value, global_mean)

def get_feature_impact(model, input_features: np.ndarray) -> Dict[str, str]:
    """Estime l'impact des différentes features sur la prédiction"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = ['mileage', 'brand', 'model', 'origin', 'fiscal_power', 
                            'condition', 'year', 'fuel_type', 'region', 'gearbox']
            
            top_features = sorted(zip(feature_names, importances), 
                                 key=lambda x: x[1], reverse=True)[:3]
            
            impacts = {}
            for feature, importance in top_features:
                if feature == 'year':
                    impacts['year_impact'] = "élevé" if importance > 0.2 else "modéré"
                elif feature == 'mileage':
                    impacts['mileage_impact'] = "élevé" if importance > 0.15 else "modéré"
                elif feature == 'condition':
                    impacts['condition_impact'] = "important" if importance > 0.1 else "secondaire"
            
            return impacts
    except:
        pass
    return {}

@app.post("/price_prediction")
async def predict(request: Request):
    data = await request.json()
    
    try:
        # Extraction et encodage des features
        mileage = float(data["mileage"])
        fiscal_power = float(data["fiscal_power"])
        year = float(data["year"])

        # Encodage des variables catégorielles
        gearbox = gearbox_encoder.transform([data["gearbox"]])[0]
        condition = int(data["condition"])
        fuel_type = fuel_encoder.transform([data["fuel_type"]])[0]
        region = region_encoder.transform([data["region"]])[0]

        # Target encoding
        brand_encoded = target_encode_smooth_deploy('brand', data["brand"], 
                                                  params_encoder['global_mean'], 
                                                  params_encoder['mappings'])
        model_encoded = target_encode_smooth_deploy('model', data["model"], 
                                                  params_encoder['global_mean'], 
                                                  params_encoder['mappings'])
        origin_encoded = target_encode_smooth_deploy('origin', data["origin"], 
                                                   params_encoder['global_mean'], 
                                                   params_encoder['mappings'])

        # Combinaison des features
        input_features = np.array([
            mileage, brand_encoded, model_encoded, origin_encoded,
            fiscal_power, condition, year,
            fuel_type, region
        ], dtype=np.float32).reshape(1, -1)

        # Mise à l'échelle
        input_scaled = scaler.transform(input_features)
        input_scaled = np.concatenate([input_scaled, np.array([[gearbox]], dtype=np.float32)], axis=1)

        # Prédiction
        prediction = model.predict(input_scaled)
        
        # Préparation de la réponse avec plus d'informations
        response = {
            "prediction": prediction.tolist(),
            "confidence": 0.85,  # Valeur par défaut si le modèle ne fournit pas de proba
            **get_feature_impact(model, input_scaled)
        }

        # Ajout des probabilités si disponible
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_scaled)[0]
            confidence = np.max(probabilities)
            response.update({
                "probabilities": probabilities.tolist(),
                "confidence": float(confidence)
            })

        return response

    except Exception as e:
        return {"error": str(e), "details": "Vérifiez que tous les champs sont correctement remplis"}