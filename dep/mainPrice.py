from fastapi import FastAPI, Request
import joblib
import numpy as np
import json

# Load model and preprocessing tools
model = joblib.load("models/catboost_model.pkl")
# model = joblib.load("models/modele_cat.pkl")
# model = joblib.load("models/lgbm_model.pkl")


scaler = joblib.load("scaler/scaler.pkl")

fuel_encoder = joblib.load("encoders/label_encoder_fuel_type.joblib")
region_encoder = joblib.load("encoders/label_encoder_region.joblib")
gearbox_encoder = joblib.load("encoders/label_encoder_gearbox.joblib")

params_encoder = joblib.load("encoders/target_encoding_params.joblib")

app = FastAPI()

def target_encode_smooth_deploy(cat_col, value, global_mean, mappings):
    # Return the target encoded value using the precomputed mapping
    return mappings.get(cat_col, {}).get(value, global_mean)

@app.post("/price_prediction")
async def predict(request: Request):
    data = await request.json()
    
    try:
        # Extract and encode features
        mileage = float(data["mileage"])
        fiscal_power = float(data["fiscal_power"])
        condition = float(data["condition"])
        year = float(data["year"])

        # Encode categorical variables
        gearbox = gearbox_encoder.transform([data["gearbox"]])[0]
        fuel_type = fuel_encoder.transform([data["fuel_type"]])[0]
        region = region_encoder.transform([data["region"]])[0]

        # Target encoding for brand, model, origin
        brand_encoded = target_encode_smooth_deploy('brand', data["brand"], params_encoder['global_mean'], params_encoder['mappings'])
        model_encoded = target_encode_smooth_deploy('model', data["model"], params_encoder['global_mean'], params_encoder['mappings'])
        origin_encoded = target_encode_smooth_deploy('origin', data["origin"], params_encoder['global_mean'], params_encoder['mappings'])

        # Combine all features (without gearbox for scaling)
        input_features = np.array([
            mileage, brand_encoded, model_encoded, origin_encoded,
            fiscal_power, condition, year,
            fuel_type, region
        ], dtype=np.float32).reshape(1, -1)

        # Scale only the numeric features
        input_scaled = scaler.transform(input_features)

        # Re-attach the gearbox feature without scaling
        input_scaled = np.concatenate([input_scaled, np.array([[gearbox]], dtype=np.float32)], axis=1)

        # Predict
        prediction = model.predict(input_scaled)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        return {"error": str(e)}
