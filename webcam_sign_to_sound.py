import os
import cv2
import torch
import numpy as np
import pyttsx3
import threading
import mediapipe as mp
from model import SignLSTM
from mediapipe_extractor import MediaPipeFeatureExtractor

# 1. Setup paths and device
DATASET_PATH = r"D:\signtosound\data\INDIAN_SIGN_LANGUAGE_NUMPY_ARRAY_SKELETAL_POINT_DATASET\data"
MODEL_PATH = "isl_model.pth"
device = torch.device("cpu") 

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 2. Load Model & Labels
labels = sorted(os.listdir(DATASET_PATH))
model = SignLSTM(len(labels)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

extractor = MediaPipeFeatureExtractor()

# Robust Audio Threading
def speak(text):
    def _thread():
        try:
            local_engine = pyttsx3.init()
            local_engine.say(text)
            local_engine.runAndWait()
            local_engine.stop()
        except: pass
    threading.Thread(target=_thread, daemon=True).start()

cap = cv2.VideoCapture(0)
sequence = []
predictions_buffer = [] 
last_spoken = ""

print("--- ISL TRANSLATOR: CONFIDENCE TARGET 0.98 ---")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) 
    black_frame = np.zeros_like(frame)
    feat = extractor.extract_features(frame)
    
    # Check if hand is detected to prevent ghosting
    hand_detected = extractor.hands_result and extractor.hands_result.multi_hand_landmarks

    if hand_detected:
        for hand_landmarks in extractor.hands_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(black_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        sequence.append(feat)
        sequence = sequence[-30:] # LSTM window size

        if len(sequence) == 30:
            x = torch.from_numpy(np.array([sequence])).float().to(device)
            with torch.no_grad():
                res = model(x)
                prob = torch.nn.functional.softmax(res, dim=1)
                confidence, pred_idx = torch.max(prob, 1)
                
                conf_val = confidence.item()
                text = labels[pred_idx.item()]

                # Terminal Monitor
                print(f"Confidence: {conf_val:.2f} | Predict: {text}", end='\r')

                # --- AUDIO LOGIC: Above 98% Confidence ONLY ---
                if conf_val >= 0.98: 
                    predictions_buffer.append(text)
                    predictions_buffer = predictions_buffer[-15:] # Stability check

                    if len(predictions_buffer) == 15 and len(set(predictions_buffer)) == 1:
                        if text != last_spoken:
                            speak(text)
                            last_spoken = text

                    # Visual feedback for high confidence
                    cv2.putText(black_frame, f"Confirmed: {text} ({conf_val*100:.1f}%)", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                else:
                    # Low confidence status
                    cv2.putText(black_frame, f"Detecting: {text}...", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    else:
        # Reset memory when hand is gone
        sequence = []
        predictions_buffer = []
        cv2.putText(black_frame, "No Hand Detected", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow("ISL Translator", black_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
