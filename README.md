# 🧠 Gait Anomaly Detection Using Deep Learning (Multi-Agent System)

This project implements a **Gait Anomaly Detection** system using multiple deep learning models in an ensemble architecture, deployed via a Flask web application. It takes in a human walking video, extracts keypoints, and detects anomalies in gait using trained models.

---

## 🚀 Demo

▶️ **Watch the demo video:** [Click to view](YOUR_VIDEO_LINK_HERE)

![Demo Screenshot](static/demo_screenshot.png)

---

## 🧐 What It Does

- Accepts a walking **video upload** through the web interface.
- Extracts **33 body keypoints per frame** using a custom keypoint extractor.
- Applies **4 deep learning models** (CNN, LSTM, GRU, Autoencoder hybrid) to analyze the motion.
- Predicts the type of **gait anomaly**:
  - Normal
  - Limping
  - Slouch
  - No Arm Swing
  - Circumduction
- Uses a **multi-agent voting mechanism** to finalize the most likely anomaly class.

---

## 🧠 Models Used

All models were trained and tested by the author on a custom dataset:

| Model Name                 | Description                   |
|---------------------------|-------------------------------|
| Autoencoder_Classifier_CNN| Feature Compression + CNN     |
| CNN_GRU                   | Convolution + GRU             |
| CNN_LSTM                  | Convolution + LSTM            |
| RNN_CNN                   | Recurrent + CNN Hybrid        |

All models are saved in `.h5` format and loaded dynamically.

---

## 📂 Dataset

This project uses a **custom dataset**:
- Curated from **YouTube walking videos**
- **Personal recorded gait videos**
- Labeled into 5 categories: Normal, Limping, Slouch, No Arm Swing, Circumduction

---

## ⚙️ How It Works

1. User uploads a walking video.
2. The system extracts **keypoints** frame-by-frame using `extract_keypoints()`.
3. The extracted data is **scaled and reshaped**.
4. Each model in the ensemble makes a prediction.
5. A **voting system** aggregates the results.
6. The app renders:
   - Final predicted gait anomaly
   - Model-wise prediction counts
   - Annotated output video

---

## 📦 Tech Stack

- **Flask** (Python Web App)
- **TensorFlow / Keras** (Model Loading)
- **MediaPipe + OpenCV** (Video Processing & Keypoint Extraction)
- **Joblib** (Scaler for preprocessing)
- **HTML5 + Jinja2** (Frontend Rendering)

---

🧭 Future Scope
With access to larger, more diverse, and medically annotated datasets, this system has strong potential for real-world clinical applications, such as:

- 🏥 Physiotherapy and rehabilitation monitoring

- 👨‍⚕️ Early diagnosis of neurological conditions (e.g., Parkinson’s, stroke-related gait issues)

- 📉 Post-surgery gait recovery analysis

- 🤖 Integration into wearables or mobile apps for home-based gait screening

- 🔬 With better data and continued fine-tuning, this multi-model system could evolve into a reliable gait analysis tool for doctors, physiotherapists, and sports scientists.

---

📌 To Do
- Add webcam real-time gait detection

- Add mobile responsiveness to frontend

- Publish dataset and training scripts

- Dockerize for cloud deployment

