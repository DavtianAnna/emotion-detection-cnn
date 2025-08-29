# Emotion Detection using CNN 

This project implements a **Convolutional Neural Network (CNN)** for predicting human emotions from facial images.  
The model is built using **Keras** with TensorFlow backend and leverages standard computer vision and machine learning libraries.  
The dataset used for training is the [**Emotion Detection FER Dataset**](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer) from Kaggle.

---

## 🚀 Features
- Emotion prediction from facial images
- Preprocessing images with **OpenCV** and **NumPy**
- Encoded labels using **scikit-learn**
- CNN architecture with **Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization**
- Training and evaluation using **Keras Sequential API**
- Visualization of accuracy and loss with **Matplotlib**
- Real-time webcam emotion detection

---

## 📂 Dataset
- Dataset includes labeled images of **7 emotions**:
  - 😡 Angry 
  - 🤢 Disgust  
  - 😨 Fear  
  - 😀 Happy  
  - 😐 Neutral  
  - 😢 Sad  
  - 😲 Surprise  
- Preprocessed to **grayscale (48x48)**  
- Split into **training** and **testing** sets

---

## 🏗️ Model Architecture
- **Convolutional layers** with ReLU activation for feature extraction  
- **MaxPooling** for dimensionality reduction  
- **Dropout layers** to reduce overfitting  
- **Dense layers** with **Softmax** output for classification into 7 emotions:
  - 😡 Angry 
  - 🤢 Disgust  
  - 😨 Fear  
  - 😀 Happy  
  - 😐 Neutral  
  - 😢 Sad  
  - 😲 Surprise 

---

## ⚙️ Training
- **Loss**: `categorical_crossentropy`  
- **Optimizer**: `Adam`  
- **Metrics**: `accuracy`  
- **Validation split**: applied to monitor generalization  
- **Epochs**: configurable (e.g., 50)  
- **Batch size**: `128`  

---

## 🚀 Results
- ✅ Validation Accuracy: **~97%**  
- 🌍Model generalizes well to unseen images  

---

## 📹 Real-Time Emotion Detection

After training, the model was exported as a **`.pkl` file**.  
A separate Python script was created (`emotion_detection.py`) to use the saved model for **real-time emotion detection** with a webcam in **VS Code**.  

---

## 🛠️ Getting Started

### Installation
```bash
git clone https://github.com/DavtianAnna/emotion-detection-cnn.git
cd emotion-detection-cnn
pip install -r requirements.txt
