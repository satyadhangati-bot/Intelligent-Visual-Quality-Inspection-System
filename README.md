# Intelligent-Visual-Quality-Inspection-System

An end-to-end computer vision system for automated quality inspection that
detects defects in manufactured products using deep learning and anomaly
detection techniques.

## Features
- CNN-based defect classification
- Autoencoder-based anomaly detection
- Transfer learning with ResNet
- Visual explainability using Grad-CAM
- Scalable training and inference pipeline

## Architecture
Image → Preprocessing → CNN / Autoencoder → Prediction → Explainability

## Tech Stack
Python, PyTorch, OpenCV, NumPy, Matplotlib,
Transfer Learning, Autoencoders, Grad-CAM

## How to Run
```bash
pip install -r requirements.txt
python src/train_classifier.py
python inference.py
