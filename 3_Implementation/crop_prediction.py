import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
import numpy as np

soil_data = pd.read_excel('/content/drive/MyDrive/sem-4AIML/soil_detection/organized_dataset/SOIL DATA GR.xlsx')
crop_data = pd.read_csv('/content/drive/MyDrive/sem-4AIML/soil_detection/organized_dataset/Crop_recommendation 2.csv')

image_model_path = '/content/drive/MyDrive/sem-4AIML/soil_detection/custom_model/final_custom_model.keras'
image_model = tf.keras.models.load_model(image_model_path)

def predict_soil_type(image_path):
    """
    Predict soil type from the input image.
    """
    # Preprocess the image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict the soil type
    soil_type_index = np.argmax(image_model.predict(image_array), axis=1)[0]
    soil_types = ['Alluvial', 'Black', 'Clay', 'Red']
    predicted_soil_type = soil_types[soil_type_index]

    return predicted_soil_type
soil_data = soil_data.rename(columns={
    'N_NO3 ppm': 'N',
    'P ppm': 'P',
    'K ppm ': 'K'
})


X_soil = soil_data[['N', 'P', 'K']]
y_soil = soil_data['pH']
ph_model = RandomForestRegressor(n_estimators=100, random_state=42)
ph_model.fit(X_soil, y_soil)


def predict_ph(N, P, K):
    return ph_model.predict([[N, P, K]])[0]


def recommend_crop(N, P, K, predicted_pH):
    crop_data['score'] = crop_data.apply(
        lambda row: abs(row['N'] - N) + abs(row['P'] - P) + abs(row['K'] - K) + abs(row['ph'] - predicted_pH),
        axis=1
    )
    best_match = crop_data.loc[crop_data['score'].idxmin()]
    return best_match['label']

def full_pipeline(image_path, N, P, K):
    soil_type = predict_soil_type(image_path)
    predicted_pH = predict_ph(N, P, K)
    recommended_crop = recommend_crop(N, P, K, predicted_pH)
    return soil_type, predicted_pH, recommended_crop

soil_type, pH, crop = full_pipeline('/content/drive/MyDrive/sem-4AIML/soil_detection/organized_dataset/test/Black/Black_10_jpg.rf.e389ebb34f43d1c63b11c7c55b285032.jpg', 50, 40, 60)
print(f"Soil: {soil_type}, pH: {pH}, Crop: {crop}")
