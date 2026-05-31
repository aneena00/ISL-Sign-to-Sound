## Sign2Sound Multimodal: Sign Language Translation Protocol

An interactive, real-time multimodal deep learning application that extracts human skeletal hand configurations and facial emotion frames via web camera to translate Indian Sign Language (ISL) gestures dynamically into continuous typed words and spoken speech strings.

 Core Features

* Real-Time Gesture Tracking: Utilizes MediaPipe to cleanly map 21 hand landmarks across dual spatial fields.
* Sequential Classification Engine: Powered by a deep Long Short-Term Memory (LSTM) network processing structural inputs over 30-frame temporal sequence windows.
* Word Building Pipeline: Integrates individual signed alphabetic characters into cohesive typed words.
* Keyboard Interfacing Control System: Integrated keyboard listeners allowing swift string management:
    * `Spacebar`: Marks the completion of a full text block, queuing the word into a background PyTTSx3 thread for speech execution and clearing the buffer canvas.
    * `Backspace`: Destroys the last appended character instantly to undo accidental or false inputs.
* Black Screen UX Layout: Blocks ambient background environments to present an optimized, distraction-free graphical viewport displaying glowing milestone skeletal topologies.
* Integrated Facial Contrast Analyzer: Dynamically estimates eye, eyebrow, and mouth variance configurations to update live user emotional state metrics concurrently with gesture processing.

 System Architecture & File Layout

* `webcam_sign_to_sound.py` - Core multi-threaded execution file managing webcam collection, structural UI rendering, and keyboard inputs.
* `model.py` - Contains the structural definition of the **SignLSTM** network layers.
* `calculate_emotion.py` - Module isolating face bounds to filter noise and output facial expressions (Happy, Sad, Surprised, Angry, Neutral).
* `mediapipe_extractor.py` - Extracted layer translating 3D coordinates into fixed feature maps.
* `requirements.txt` - File holding system dependencies.

 Dataset Specifications & Missing Label Optimization

The network evaluates geometric arrays containing standardized positional values extracted across 25 dedicated character sets (A-Z).

> **Implementation Notice:** Due to physical transition complexities, the dataset deliberately omits structural data for the letter **`R`**. The system dynamically scales to a **25-class highway configuration** array map to completely avoid label shifting or index classification mismatches.
