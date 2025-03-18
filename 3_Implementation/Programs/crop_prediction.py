import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load data
soil_data = pd.read_excel(r"C:\New folder (3)\Dataset\xlfiles\SOIL DATA GR.xlsx")
crop_data = pd.read_csv(r"C:\New folder (3)\Dataset\xlfiles\Crop_recommendation 2.csv")

# Load image model
image_model_path = r"C:\New folder (3)\custom_cnn_production_model.keras"
image_model = tf.keras.models.load_model(image_model_path, compile=False)

# Define soil types and default values
default_values = {
    "Alluvial": {"N": 50, "P": 40, "K": 60},
    "Black": {"N": 45, "P": 35, "K": 55},
    "Clay": {"N": 40, "P": 30, "K": 50},
    "Red": {"N": 55, "P": 45, "K": 65},
}
soil_types = list(default_values.keys())

# Rename columns for consistency
soil_data = soil_data.rename(columns={
    "N_NO3 ppm": "N",
    "P ppm": "P",
    "K ppm ": "K",
})

# Calculate min and max values from crop data
n_min, n_max = crop_data['N'].min(), crop_data['N'].max()
p_min, p_max = crop_data['P'].min(), crop_data['P'].max()
k_min, k_max = crop_data['K'].min(), crop_data['K'].max()

def predict_soil_type(image_path):
    """Predict soil type from an image and return confidence scores."""
    image = load_img(image_path, target_size=(128, 128))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    predictions = image_model.predict(image_array)[0]
    soil_type_index = np.argmax(predictions)
    
    print("\nSoil Type Probabilities:")
    for i, soil in enumerate(soil_types):
        print(f"{soil}: {predictions[i] * 100:.2f}%")
    
    return soil_types[soil_type_index]

def recommend_crop(N, P, K):
    """Recommend the best crop based on clipped NPK values."""
    # Clip input values to dataset ranges
    N = np.clip(N, n_min, n_max)
    P = np.clip(P, p_min, p_max)
    K = np.clip(K, k_min, k_max)
    
    crop_data["score"] = (
        abs(crop_data["N"] - N) +
        abs(crop_data["P"] - P) +
        abs(crop_data["K"] - K)
    )
    
    best_match = crop_data.loc[crop_data["score"].idxmin()]
    return best_match["label"]

def get_user_input(soil_type):
    """Improved input validation with error handling"""
    print(f"\nEnter values for Nitrogen (N), Phosphorus (P), and Potassium (K) OR press Enter to use default values for {soil_type} soil.")

    def get_value(element, default):
        while True:
            user_input = input(f"Enter {element} (default: {default}): ").strip()
            if not user_input:
                return default
            try:
                return float(user_input)
            except ValueError:
                print("Invalid input! Please enter a valid number.")

    N = get_value("N", default_values[soil_type]["N"])
    P = get_value("P", default_values[soil_type]["P"])
    K = get_value("K", default_values[soil_type]["K"])
    
    return N, P, K

def full_pipeline(image_path):
    """Complete pipeline to detect soil and recommend crops"""
    soil_type = predict_soil_type(image_path)
    print(f"\n🟢 **Detected Soil Type:** {soil_type}")

    N, P, K = get_user_input(soil_type)
    recommended_crop = recommend_crop(N, P, K)
    return soil_type, recommended_crop

# Example Usage
image_path = r"C:\New folder (3)\Dataset\test\Clay soil\Clay_11.jpg"  
soil_type, crop = full_pipeline(image_path)
print(f"\n✅ **Final Output:** Soil: {soil_type}, Recommended Crop: {crop}")
