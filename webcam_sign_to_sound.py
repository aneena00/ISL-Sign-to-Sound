import os
import cv2
import torch
import numpy as np
import pyttsx3
import threading
import queue
import time
import mediapipe as mp

# Force Python 3.13 to link the solutions modules dynamically
if not hasattr(mp, 'solutions'):
    import mediapipe.python.solutions as solutions
    mp.solutions = solutions

from model import SignLSTM
from mediapipe_extractor import MediaPipeFeatureExtractor

# --- EMOTION PIPELINE INTEGRATION ---
try:
    import calculate_emotion
except ImportError:
    calculate_emotion = None
    print("Warning: 'calculate_emotion.py' module could not be found.")

# 1. Configuration
DATASET_PATH = r"D:\signtosound\data\INDIAN_SIGN_LANGUAGE_NUMPY_ARRAY_SKELETAL_POINT_DATASET\data"
MODEL_PATH = "isl_model.pth"
device = torch.device("cpu") 

# Dynamically read your 25 folders (Automatically accounts for the missing 'R')
labels = sorted(os.listdir(DATASET_PATH))

model = SignLSTM(len(labels)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- WORD BUILDING BUFFER CONFIGURATION ---
current_word = ""          # Stores your accumulated word string
last_added_letter = ""     # Prevents the same letter from duplicate spamming
last_addition_time = 0     # Timestamp tracking for typing delays
DEBOUNCE_DELAY = 1.5       # Time delay (seconds) required before re-typing the same letter
current_emotion = "Neutral" 

# 2. Audio Engine Thread Queue
audio_queue = queue.Queue()

def say_text():
    while True:
        text = audio_queue.get()
        if text is None: break
        
        try:
            print(f"--- SPEAKING WORD: {text} ---")
            temp_engine = pyttsx3.init('sapi5')
            temp_engine.say(text)
            temp_engine.runAndWait()
            del temp_engine
        except Exception as e:
            print(f"Speech error: {e}")
        finally:
            audio_queue.task_done()

threading.Thread(target=say_text, daemon=True).start()

# 3. Main Webcam Tracking Loop
extractor = MediaPipeFeatureExtractor()
cap = cv2.VideoCapture(0)

sequence = []
stability_buffer = []

print("\n=== BLACK SCREEN WORD INTEGRATOR & EMOTION PIPELINE ONLINE ===")
print("Keyboard Controls:")
print("-> [SPACEBAR]  : Speak and clear the accumulated word buffer")
print("-> [BACKSPACE] : Erase the last signed letter from your word")
print("-> [Q]         : Close application safely\n")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Create the complete black screen layout canvas
    black_frame = np.zeros_like(frame)
    
    # --- LIVE REAL-TIME EMOTION DETECTION ---
    if calculate_emotion is not None:
        try:
            # Feeds your live webcam frame straight into your face processing function
            current_emotion = calculate_emotion.calculate_emotion(frame)
        except Exception as e:
            current_emotion = "Error reading"
    else:
        current_emotion = "Neutral (No Script)"

    features = extractor.extract_features(frame)
    
    # Draw tracking skeletons directly on the black frame
    if extractor.hands_result and extractor.hands_result.multi_hand_landmarks:
        for hand_landmarks in extractor.hands_result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                black_frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )
            
        sequence.append(features)
        sequence = sequence[-30:] 

        if len(sequence) == 30:
            with torch.no_grad():
                res = model(torch.from_numpy(np.array([sequence])).float().to(device))
                prob = torch.nn.functional.softmax(res, dim=1)
                max_prob, idx = torch.max(prob, dim=1)
                
                prediction = labels[idx.item()]
                confidence = max_prob.item()

                stability_buffer.append(prediction)
                stability_buffer = stability_buffer[-5:] 

                # If the prediction passes your confidence filter thresholds
                if confidence >= 0.80 and stability_buffer.count(prediction) >= 3:
                    status_text = f"Confirmed: {prediction} ({confidence:.2f})"
                    color = (0, 255, 0)
                    
                    # --- WORD BUILDING LOGIC ENGINE ---
                    current_time = time.time()
                    if prediction != last_added_letter or (current_time - last_addition_time > DEBOUNCE_DELAY):
                        current_word += prediction
                        print(f"Added Letter: {prediction} | Word Buffer: {current_word}")
                        last_added_letter = prediction
                        last_addition_time = current_time
                else:
                    status_text = f"Detecting: {prediction} ({confidence:.2f})"
                    color = (0, 255, 255)

                cv2.putText(black_frame, status_text, (50, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    else:
        sequence = []
        stability_buffer = []
        last_added_letter = ""  # Reset tracking state when hands leave frame space
        cv2.putText(black_frame, "Searching for Hands...", (50, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # --- UI DISPLAY HUD CANVAS PANEL ---
    # Draw a clean dark bottom bar
    cv2.rectangle(black_frame, (0, h - 100), (w, h), (20, 20, 20), -1)
    
    # Overlay the typed word construction string
    cv2.putText(black_frame, f"Word Buffer: {current_word}", (30, h - 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    
    # Overlay the calculated facial expression metrics
    cv2.putText(black_frame, f"Emotion: {current_emotion}", (30, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
    # --- DIAGNOSTIC CALIBRATION OVERLAY ---
    
    cv2.imshow("Sign2Sound Multimodal", black_frame)
    
    # --- KEYBOARD TRIGGER MAP CONTROLLER ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == 32:  # Spacebar key hit
        if current_word:
            audio_queue.put(current_word)
            current_word = ""  # Reset the word buffer for your next phrase
            last_added_letter = ""
    elif key == 8:   # Backspace key hit
        current_word = current_word[:-1]
        print(f"Erased last letter. Current buffer: {current_word}")

audio_queue.put(None)
cap.release()
cv2.destroyAllWindows()
