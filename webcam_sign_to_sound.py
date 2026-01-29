import os
import cv2
import torch
import numpy as np
import pyttsx3
import threading
import queue
from model import SignLSTM
from mediapipe_extractor import MediaPipeFeatureExtractor

# 1. Setup & Device
DATASET_PATH = r"D:\signtosound\data\INDIAN_SIGN_LANGUAGE_NUMPY_ARRAY_SKELETAL_POINT_DATASET\data"
MODEL_PATH = "isl_model.pth"
device = torch.device("cpu") 

labels = sorted(os.listdir(DATASET_PATH))
model = SignLSTM(len(labels)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

extractor = MediaPipeFeatureExtractor()
engine = pyttsx3.init()
audio_queue = queue.Queue()

def audio_worker():
    while True:
        text = audio_queue.get()
        if text is None: break
        engine.say(text)
        engine.runAndWait()
        audio_queue.task_done()

threading.Thread(target=audio_worker, daemon=True).start()

# --- FULL SCREEN SETUP ---
WINDOW_NAME = "ISL Translator"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)
sequence = []
sentence = []
last_spoken = ""
stability_buffer = []

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    # Create black frame for skeleton
    black_frame = np.zeros_like(frame)
    feat = extractor.extract_features(frame)
    
    if extractor.hands_result and extractor.hands_result.multi_hand_landmarks:
        # Draw landmarks on the black frame
        for hand_landmarks in extractor.hands_result.multi_hand_landmarks:
            import mediapipe as mp
            mp.solutions.drawing_utils.draw_landmarks(
                black_frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        
        sequence.append(feat)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            x = torch.from_numpy(np.array([sequence])).float().to(device)
            with torch.no_grad():
                res = model(x)
                prob = torch.nn.functional.softmax(res, dim=1)
                max_prob, idx = torch.max(prob, dim=1)
                
                prediction = labels[idx.item()]
                confidence = max_prob.item()

                stability_buffer.append(prediction)
                stability_buffer = stability_buffer[-15:]

                # Only speak if confidence is high and stable
                if confidence >= 0.98 and stability_buffer.count(prediction) == 15:
                    status_text = f"Confirmed: {prediction} ({confidence:.2f})"
                    color = (0, 255, 0)
                    if prediction != last_spoken:
                        audio_queue.put(prediction)
                        last_spoken = prediction
                else:
                    status_text = f"Detecting: {prediction} ({confidence:.2f})"
                    color = (0, 255, 255)

                cv2.putText(black_frame, status_text, (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        sequence = []
        stability_buffer = []
        cv2.putText(black_frame, "No Hand Detected", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the processed black frame in the full-screen window
    cv2.imshow(WINDOW_NAME, black_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

audio_queue.put(None)
cap.release()
cv2.destroyAllWindows()
