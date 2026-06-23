# 🧠 Handwritten Digit Recognition System

A Deep Learning-based Handwritten Digit Recognition System built using TensorFlow, Keras, OpenCV, and Streamlit.

The application allows users to draw handwritten digits directly on an interactive canvas and predicts the digit using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

---

## 🚀 Features

* CNN-based digit classification
* Trained on 70,000 handwritten digit images
* Interactive drawing canvas
* Real-time digit prediction
* Confidence score display
* OpenCV image preprocessing
* Streamlit web application

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Streamlit
* Streamlit Drawable Canvas

---

## 📊 Dataset

The model is trained using the MNIST dataset:

* 60,000 training images
* 10,000 testing images
* 10 digit classes (0–9)
* Image size: 28 × 28 pixels

---

## 🧠 Model Architecture

CNN Architecture:

Input Layer (28×28×1)

↓ Conv2D (32 Filters)

↓ MaxPooling2D

↓ Conv2D (64 Filters)

↓ MaxPooling2D

↓ Flatten

↓ Dense (128)

↓ Dropout (0.3)

↓ Dense (10 Softmax Outputs)

---

## 📈 Results

* Test Accuracy: ~98–99%
* Real-time digit prediction
* Confidence-based classification

---

## 🎯 Application Workflow

User Draws Digit

↓
OpenCV Preprocessing

↓
CNN Prediction

↓
Digit Classification

↓
Confidence Score Display

---

<h2>Application Demo</h2>

<p align="center">
  <img src="assets/app_home.png" width="400">
  <img src="assets/digit_prediction.png" width="400">
</p>

<p align="center">
  <img src="assets/processed_digit.png" width="400">
</p>

---

## Features

- CNN-based digit recognition
- Interactive drawing canvas
- OpenCV preprocessing
- Real-time predictions
- Confidence score display

## 📁 Project Structure

```text
Handwritten-Digit-Recognition/
│
├── assets/
│   ├── accuracy_plot.png
│   ├── confusion_matrix.png
│   ├── loss_plot.png
│   └── sample_digits.png
│
├── data/
├── models/
│   └── digit_recognition_model.h5
│
├── notebooks/
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/harshitha121124/handwritten-digit-recognition.git

cd handwritten-digit-recognition

pip install -r requirements.txt

streamlit run app.py
```

## 🔮 Future Improvements

* Support for uploaded handwritten images
* Multi-digit recognition
* EMNIST alphabet recognition
* Mobile deployment
* Improved real-world handwriting support

---

## 👩‍💻 Author

**G. R. Harshitha**

B.Tech Mechanical Engineering
Indian Institute of Technology Madras

---

⭐ If you found this project useful, consider giving it a star.

