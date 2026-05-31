import cv2
import numpy as np

# Load the built-in OpenCV Face bounding box scanner
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables to pass numbers to your main display screen for debugging
latest_mouth_var = 0.0
latest_brow_var = 0.0

def calculate_emotion(frame):
    """
    Scans the live webcam frame, extracts real-time facial contrast metrics,
    and updates global variables so we can calibrate thresholds live.
    """
    global latest_mouth_var, latest_brow_var
    
    if frame is None:
        return "Neutral"

    try:
        # 1. Prepare image frame layout
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
        
        if len(faces) == 0:
            latest_mouth_var = 0.0
            latest_brow_var = 0.0
            return "Neutral"
            
        # 2. Target the primary user's face boundaries
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        # 3. Analyze Facial Feature Zones (Eyes, Brows, and Mouth Layout)
        mouth_zone = roi_gray[int(h*0.65):int(h*0.95), int(w*0.2):int(w*0.8)]
        eye_brow_zone = roi_gray[int(h*0.1):int(h*0.45), int(w*0.15):int(w*0.85)]
        
        # Calculate structural variance (live intensity changes)
        latest_mouth_var = float(np.var(mouth_zone)) if mouth_zone.size > 0 else 0.0
        latest_brow_var = float(np.var(eye_brow_zone)) if eye_brow_zone.size > 0 else 0.0
        
        # 4. Calibration-Ready Decision Logic
        # (We will use wide, easy numbers first so it responds to your expressions)
        # 4. Calibration-Ready Decision Logic
        # We raise the safety bar so your normal face numbers don't trigger "Surprise"
        if latest_mouth_var > 1100 and latest_brow_var > 1300:
            return "Surprise"
        elif latest_mouth_var > 750:
            return "Happy"
        elif latest_mouth_var < 400 and latest_brow_var > 800:
            return "Angry"
        elif latest_mouth_var < 300 and latest_brow_var < 300:
            return "Sad"
        else:
            return "Neutral"

    except Exception as e:
        return "Neutral"
