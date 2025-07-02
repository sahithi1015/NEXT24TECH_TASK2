# NEXT24TECH_TASK2
AIML Internship

# 🌿 Plant Leaf Disease Detection System Using AI

This project is an AI-powered system that detects plant leaf diseases using deep learning, built to support early diagnosis and sustainable agriculture. It helps farmers identify diseases in crops like tomato, potato, and bell pepper by uploading an image of a leaf and receiving instant feedback.

---

## 📌 Project Objective

The goal of this project is to:
- Detect plant diseases from leaf images using AI algorithms
- Improve agricultural yield and quality through early detection
- Build a user-friendly tool for farmers using a Streamlit web interface

---

## 🗂 Dataset

We used the publicly available [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) which contains over *40,000+ images* of healthy and diseased plant leaves across *15+ classes*.

- Total Images: 41,000+
- Number of Classes: 15 (Tomato, Potato, Pepper, etc.)
- Image Types: JPG
- Split: 80% Training, 20% Validation

---

## 🤖 Model Architecture

- Model: Transfer Learning with *MobileNetV2*
- Input Shape: 128x128x3
- Output: Softmax layer for multiclass disease classification
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy

### ✅ Training Summary
- Training Accuracy: ~56%
- Final Validation Accuracy: ~85% (after fine-tuning)
- Epochs: 10–15

---

## 🌐 Streamlit Web App

A simple web interface built using *Streamlit* that allows users to upload leaf images and get real-time predictions.

### Features:
- Upload .jpg, .jpeg, .png images
- Get predicted disease class
- See prediction confidence
- Mobile-friendly interface

---

## 🛠 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/sahithi1015/plant-disease-detector.git
cd plant-disease-detector
