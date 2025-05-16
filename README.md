# Emotion-Detection-Using-Speech-and-Facial-Recognition.

**Features**

- Detects emotions from facial images using a lightweight CNN (Mini-XCEPTION).
- Recognizes emotions from speech using audio feature extraction and a support vector machine (SVM) classifier.
- Integrates OpenAI’s Whisper model for robust, multilingual speech-to-text transcription.
- Interactive dashboard for uploading images or audio and displaying results.

---

**System Overview**

### Facial Emotion Recognition

- **Model**: Pre-trained Mini-XCEPTION CNN, trained on the FER2013 dataset.
- **Pipeline**:
  - Input image is converted to grayscale, resized to 64×64, and normalized.
  - OpenCV detects faces; each face region is processed by the CNN.
  - Output is one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
- **Libraries**: Keras, OpenCV, NumPy, Streamlit (optional for UI).
- **Performance**: ~66% accuracy on FER2013 test set.

### Speech Emotion Recognition

- **Model**: SVM classifier trained on synthetic audio features (MFCC, pitch).
- **Speech-to-Text**: Whisper LLM for transcription and language detection.
- **Pipeline**:
  - Audio is converted to mono WAV, resampled to 16kHz.
  - MFCC and pitch features are extracted using librosa.
  - Features are classified by the SVM into emotions (happy, sad, angry, neutral).
- **Libraries**: Whisper, librosa, scikit-learn, NumPy, pydub.

---

**User Interface**

- Built with Streamlit for easy interaction.
- Users can upload images or audio files.
- Outputs include:
  - Annotated images with emotion labels.
  - Transcribed speech, detected language, and predicted emotion.

---

**Installation**

Install required dependencies:
```bash
pip install keras opencv-python openai-whisper librosa==0.10.0.post2 scikit-learn numpy==1.23.5 pydub ffmpeg-python streamlit
```

---

**Usage**

1. **Facial Emotion Detection**
   - Upload an image.
   - The system detects faces, classifies emotions, and displays results.

2. **Speech Emotion Detection**
   - Upload an audio file.
   - The system transcribes speech, detects language, extracts features, and predicts emotion.

---

**Limitations**

- Relies on frontal face detection; sensitive to lighting, occlusions, and pose.
- Audio classifier is based on synthetic labels for demonstration.
- May misclassify subtle or mixed emotions.
- Does not interpret emotional context (e.g., sarcasm).

---

**Future Enhancements**

- Integrate multimodal emotion recognition (combine audio, visual, and text).
- Replace CNN with Vision Transformers (ViT) for improved facial analysis.
- Use real-time video streams and temporal emotion tracking.
- Expand dataset diversity to reduce bias.

---

**Conclusion**

This project demonstrates a practical approach to emotion detection using both facial images and speech. It provides a foundation for building emotion-aware applications, with extensibility for future improvements in multimodal AI.
