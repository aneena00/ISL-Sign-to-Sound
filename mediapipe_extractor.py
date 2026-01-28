import cv2
import numpy as np
import mediapipe as mp

class MediaPipeFeatureExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        # max_num_hands=2 is vital for ISL dataset alignment
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands_result = None
        # TOGGLE: Set to True if your physical Right hand shows as 'Left' in terminal
        self.SWAP_HANDS = True

    def extract_features(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.hands_result = self.hands.process(rgb)
        
        # Initialize empty 63-point vectors (21 landmarks * 3 coordinates)
        left_hand = np.zeros(63)
        right_hand = np.zeros(63)

        if self.hands_result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(self.hands_result.multi_hand_landmarks, self.hands_result.multi_handedness):
                
                # Wrist Normalization: Landmark 0 is the wrist
                wrist = hand_landmarks.landmark[0]
                lm_list = []
                for lm in hand_landmarks.landmark:
                    lm_list.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                
                lm_array = np.array(lm_list).flatten()
                label = handedness.classification[0].label 

                if self.SWAP_HANDS:
                    label = 'Right' if label == 'Left' else 'Left'

                if label == 'Left':
                    left_hand = lm_array
                else:
                    right_hand = lm_array

        # Returns the full 126-feature vector required by SignLSTM
        return np.concatenate([left_hand, right_hand]).astype(np.float32)

    def close(self):
        self.hands.close()
