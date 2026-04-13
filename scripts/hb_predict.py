import joblib
import numpy as np
import os

def load_hb_model(model_path):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        print(f"File model tidak ditemukan")
        return None

def predict_hemoglobin(model, color_features, age, gender):
    # SUSUNAN ARRAY HARUS SAMA PERSIS DENGAN SAAT TRAINING (11 Fitur)
    # urutan - Age, Gender, mean_R, mean_G, mean_B, mean_L, mean_a, mean_b, mean_H, mean_S, mean_V
    features_array = np.array([[
        age, 
        gender,
        color_features['mean_R'], color_features['mean_G'], color_features['mean_B'],
        color_features['mean_L'], color_features['mean_a'], color_features['mean_b'],
        color_features['mean_H'], color_features['mean_S'], color_features['mean_V'],
    ]])
    
    if model is None:
        return 0.0
    
    # Melakukan prediksi 
    prediction = model.predict(features_array)
    
    return float(prediction[0])