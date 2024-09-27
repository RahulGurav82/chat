from flask import Flask, request, jsonify
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)
model = joblib.load('siren_detection_model.pkl')  # Load your pre-trained model

def extract_features_from_audio(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
    combined_features = np.hstack([mfccs_mean, spectral_contrast_mean])
    return combined_features

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    audio, sr = librosa.load(file)
    features = extract_features_from_audio(audio, sr)
    features = np.expand_dims(features, axis=0)
    prediction_prob = model.predict_proba(features)[0][1]
    return jsonify({'prediction': prediction_prob})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
