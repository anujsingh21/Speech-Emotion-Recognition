from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
import librosa
from sklearn.neural_network import MLPClassifier


app = Flask( __name__ )

# Load the pre-trained model
Pkl_Filename = "Emotion_Voice_Detection_Model.pkl"
with open(Pkl_Filename, 'rb') as file:  
    Emotion_Voice_Detection_Model = pickle.load(file)

# Home route to render the index.html page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file uploads and predictions
@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)
    
    try:
        # Extract features from the audio file
        y, sr = librosa.load(file_path, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

        # Prepare the input for prediction
        features = np.array([mfccs])
        prediction = Emotion_Voice_Detection_Model.predict(features)

        # Map prediction to emotion label
        emotion = prediction[0]  # Assuming your model returns labels directly
        os.remove(file_path)  # Clean up the uploaded file

        return jsonify({'emotion': emotion})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '_main_':
    app.run(debug=True)