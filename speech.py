import streamlit as st
import librosa
import soundfile
import numpy as np
import joblib
import tempfile
import os
import webbrowser

# Function to extract features from a sound file
def extract_features(file_name, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

# Load the model
model = joblib.load('modelForPrediction.sav')

# Define observed emotions
observed_emotions = ['calm', 'happy', 'fearful', 'disgust', 'surprised', 'angry', 'sad']
# Function to predict emotion from features
def predict_emotion(features):
    features = features.reshape(1, -1)
    emotion_id = model.predict(features)
    print("Emotion ID:", emotion_id)
    return emotion_id[0]
 
def recommend_speech_based():
    st.write("Upload a voice file and we'll predict the emotion!")

    # File upload
    uploaded_file = st.file_uploader("Choose a voice file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file)

        # Predict emotion on button click
        if st.button("Recommend Songs"):
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            # Extract features
            features = extract_features(temp_file_path)
            # Call predict function
            emotion = predict_emotion(features)
            st.write("Detected Emotion:", emotion)
            if emotion=="angry":
                 webbrowser.open(f"https://www.youtube.com/results?search_query=telugu+sid+sriram+melody+music+songs")
            elif emotion=="calm":
                 webbrowser.open(f"https://www.youtube.com/results?search_query=telugu+sid+sriram+soothing+songs")
            elif emotion=="disgust":
                 webbrowser.open(f"https://www.youtube.com/results?search_query=telugu+sid+sriram+happy and rock+songs")
            elif emotion=="happy":
                 webbrowser.open(f"https://www.youtube.com/results?search_query=telugu+sid+sriram+happy+songs")
            elif emotion=="fearful":
                 webbrowser.open(f"https://www.youtube.com/results?search_query=telugu+sid+sriram+classical+songs")
            elif emotion=="sad":
                webbrowser.open(f"https://www.youtube.com/results?search_query=telugu+sid+sriram+rock+songs")
            elif emotion=="suprise":
                webbrowser.open(f"https://www.youtube.com/results?search_query=telugu+sid+sriram+upbeat+songs+songs")
            # Delete the temporary file
            os.unlink(temp_file_path)
            print(temp_file_path)
            
recommend_speech_based()
