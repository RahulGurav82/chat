import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import sounddevice as sd
import soundfile as sf
import time
import pickle  # Import pickle for saving the model

# Function to extract features from audio
def extract_features_from_audio(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
    combined_features = np.hstack([mfccs_mean, spectral_contrast_mean])
    return combined_features

# Directory paths for siren and non-siren sounds
siren_dir = 'c:/Users/rahul/Downloads/AmbulanceAwareness/Siren_Detection/siren_sounds'
non_siren_dir = 'c:/Users/rahul/Downloads/AmbulanceAwareness/Siren_Detection/non_siren_sounds'

# Prepare the dataset
X = []
y = []

# Load and process siren sounds
for file_name in os.listdir(siren_dir):
    file_path = os.path.join(siren_dir, file_name)
    audio, sr = librosa.load(file_path)
    features = extract_features_from_audio(audio, sr)
    X.append(features)
    y.append(1)  # Label for siren sounds

# Load and process non-siren sounds
for file_name in os.listdir(non_siren_dir):
    file_path = os.path.join(non_siren_dir, file_name)
    audio, sr = librosa.load(file_path)
    features = extract_features_from_audio(audio, sr)
    X.append(features)
    y.append(0)  # Label for non-siren sounds

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# Save the model to a file with a .pkl extension
model_filename = 'siren_detection_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)
print(f"Model saved to {model_filename}")

# Continuous real-time prediction function
def continuous_real_time_prediction(model, duration=2, sample_rate=22050, threshold=0.6):
    print("Starting continuous listening for sirens...")
    while True:
        print("Listening...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait until the recording is finished
        
        # Save recorded audio for debugging
        sf.write('recorded_audio.wav', audio, sample_rate)
        
        audio = np.squeeze(audio)  # Remove single-dimensional entries from the shape of an array
        features = extract_features_from_audio(audio, sample_rate)
        features = np.expand_dims(features, axis=0)  # Reshape for prediction
        prediction_prob = model.predict_proba(features)[0][1]  # Probability of siren class
        
        print(f"Prediction Probability: {prediction_prob:.2f}")  # Print prediction probability
        
        if prediction_prob > threshold:
            print("Siren Detected!")
        else:
            print("No Siren Detected")
        
        time.sleep(5)  # Pause before the next recording

# Start the siren detection
continuous_real_time_prediction(model)
