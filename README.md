# NEXT24TECH_TASK2
AIML Internship

# 🌿 Plant Leaf Disease Detection System Using AI

This is a machine learning project that detects plant leaf diseases using a deep learning model trained on the PlantVillage dataset. The system is accessible via a simple Streamlit-based web app, where farmers or users can upload leaf images to receive real-time predictions.

---

## 📌 Objective

To build an AI-based system for detecting plant diseases at an early stage using image classification. This helps in minimizing crop damage and increasing agricultural productivity.

---

## 📂 Dataset

- *Source*: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- *Total Images*: ~41,000
- *Number of Classes*: 15 (e.g., Tomato_Bacterial_spot, Potato_Late_blight, etc.)
- *Format*: JPG images
- *Split*: 80% Training / 20% Validation using ImageDataGenerator

---

## 🤖 Model Details

- *Model Used*: MobileNetV2 (Transfer Learning)
- *Libraries*: TensorFlow, Keras
- *Input Shape*: 128x128x3
- *Architecture*:
  - Base: MobileNetV2 (frozen initially, then fine-tuned)
  - Classifier: GlobalAveragePooling + Dense layers + Dropout
- *Output*: Multiclass Softmax (15 disease classes)

---

## 🌐 Streamlit Web App

The app allows users to:

- Upload .jpg, .png, or .jpeg leaf images
- Instantly get the predicted disease
- View model confidence score

---

## 🚀 Getting Started

### 1. Clone the Repo

git clone https://github.com/sahithi1015/NEXT24TECH_TASK2.git
cd NEXT24TECH_TASK2

### 2.Install Dependencies

pip install -r requirements.txt

### 3.Run the streamlit App

streamlit run plant_disease_detector_view.py

