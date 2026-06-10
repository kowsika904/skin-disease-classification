


# 🧠 Skin Disease Classification Using Deep Learning

---

## 🌟 About The Project
This project is a Flask-based web application that uses Deep Learning (CNN model) to classify different types of skin diseases from uploaded images.

The system helps in early detection of skin conditions and supports medical diagnosis using Artificial Intelligence.

It combines:
- Deep Learning for image classification
- Flask for web development
- Medical image processing

---

## ✨ Key Features
- Upload skin image and get prediction instantly  
- AI-powered disease classification  
- Real-time prediction system  
- User-friendly Flask web interface  
- CNN-based Deep Learning model  
- Fast and efficient processing  
- Medical image analysis support  

---

1. Dataset Collection (HAM10000 dataset)
2. Data Preprocessing
   - Image resizing
   - Normalization
   - Label encoding

3. Model Building
   - CNN (Convolutional Neural Network)
   - Multiple convolution layers
   - Pooling layers

4. Model Training
   - Training on labeled dataset
   - Optimization using loss function

5. Model Evaluation
   - Accuracy check
   - Loss validation

6. Model Saving
   - Save trained model (.h5 or .npy)

7. Flask Web Application
   - Frontend UI (HTML/CSS)
   - Backend API (Flask)

8. Prediction Phase
   - User uploads image
   - Model predicts disease
   - Result displayed instantly
  
   - 

## 🏗️ System Architecture

The system follows a simple Deep Learning web application pipeline:

```text
User Uploads Image
        ↓
Flask Web Interface (Frontend)
        ↓
Flask Backend (Python Server)
        ↓
Image Preprocessing Module
        ↓
Trained CNN Deep Learning Model
        ↓
Prediction Engine
        ↓
Result Display on Web Page



1. Dataset Collection (HAM10000 Dataset)
        ↓
2. Data Preprocessing
   - Image resizing
   - Normalization
   - Label encoding
        ↓
3. Model Building
   - CNN (Convolutional Neural Network)
   - Convolution layers
   - Pooling layers
        ↓
4. Model Training
   - Training with labeled dataset
   - Optimization using loss function
        ↓
5. Model Evaluation
   - Accuracy measurement
   - Loss validation
        ↓
6. Model Saving
   - Save trained model (.h5 / .npy)
        ↓
7. Flask Web Application
   - Frontend (HTML/CSS)
   - Backend (Flask API)
        ↓
8. Prediction Phase
   - User uploads image
   - Model predicts disease
   - Result displayed on UI


---
# Clone repository
git clone https://github.com/yourusername/skin-disease-classification.git

# Navigate to project folder
cd skin-disease-classification

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py

# Open browser
http://127.0.0.1:5000/
