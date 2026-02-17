import streamlit as st
import cv2
import torch
import numpy as np
import mediapipe as mp
import pyttsx3
import threading

# --- Environment Fix ---
if not hasattr(mp, 'solutions'):
    from mediapipe.python import solutions
    mp.solutions = solutions

from model import SignLSTM
from mediapipe_extractor import MediaPipeFeatureExtractor

# --- AUDIO THREADING ---
def speak(text):
    def run_speech():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech).start()

# --- MAKE SCREEN BIG (CSS) ---
st.set_page_config(page_title="Sign to Sound", layout="wide")
st.markdown("""
    <style>
    .stMainBlockContainer {max-width: 100% !important; padding: 1rem !important;}
    div[data-testid="stImage"] img { width: 100% !important; height: auto !important; border: 2px solid #4CAF50;}
    </style>
    """, unsafe_allow_html=True)

st.title("🤟 ISL Full-Screen Translator")

@st.cache_resource
def load_assets():
    m = SignLSTM(num_classes=25)
    m.load_state_dict(torch.load("isl_model.pth", map_location="cpu"))
    m.eval()
    return m, MediaPipeFeatureExtractor()

model, extractor = load_assets()
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXY")

cap = cv2.VideoCapture(0)
# Set high resolution for better landmark detection
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_placeholder = st.empty()
prediction_text = st.empty()
sequence = []

if 'last_letter' not in st.session_state:
    st.session_state.last_letter = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    black_screen = np.zeros(frame.shape, dtype=np.uint8)
    features = extractor.extract_features(frame)
    
    if extractor.hands_result and extractor.hands_result.multi_hand_landmarks:
        for hand_landmarks in extractor.hands_result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                black_screen, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )
    
    # --- PREDICTION IMPROVEMENTS ---
    if np.any(features):
        sequence.append(features)
        sequence = sequence[-30:] # LSTM window size
        
        if len(sequence) == 30:
            with torch.no_grad():
                # Prediction logic
                out = model(torch.tensor(np.array([sequence])).float())
                prob, idx = torch.max(torch.softmax(out, dim=1), dim=1)
                
                confidence = prob.item()
                letter = labels[idx.item()]
                
                # Dynamic UI update
                prediction_text.markdown(f"<h1 style='text-align: center; color: green;'>Detected: {letter} | Confidence: {confidence:.2f}</h1>", unsafe_allow_html=True)
                
                # Stable Prediction: Only speak if confidence is high
                if confidence > 0.90 and letter != st.session_state.last_letter:
                    speak(letter)
                    st.session_state.last_letter = letter
    else:
        # Clear sequence if hands are lost to prevent "ghost" predictions
        sequence = []

    frame_placeholder.image(cv2.cvtColor(black_screen, cv2.COLOR_BGR2RGB), use_container_width=True)
