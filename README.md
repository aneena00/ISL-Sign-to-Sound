# Indian Sign Language (ISL) to Sound Converter

An AI-powered system that recognizes Indian Sign Language (ISL) hand gestures in real-time and converts them into audible speech using Deep Learning (LSTM) and MediaPipe.

->Features
- **Real-time Recognition**: Detects 26 ISL alphabetical gestures with 97%+ accuracy.
- **Two-Handed Support**: Uses a 126-feature vector to track both hands simultaneously.
- **Voice Output**: Instant text-to-speech conversion using the Windows SAPI5 driver.
- **Full-Screen UI**: Clean, distraction-free black-frame interface for better focus.

->Setup Instructions
1. **Clone the Repo**: `git clone https://github.com/aneena00/ISL-Sign-to-Sound.git`
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Run the Application**: `python webcam_sign_to_sound.py`

->Model Performance
- **Accuracy**: 97.33%
- **Confidence Threshold**: 90% (Customizable)
- **Frameworks**: PyTorch, MediaPipe, OpenCV

->Group Members
- Abhinav Bijoy
- Jan Zameera
- M K Nandana
