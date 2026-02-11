## AgriSmart – AI Powered Smart Farming Assistant

AgriSmart is a web-based smart agriculture platform designed to assist farmers in making better decisions using Artificial Intelligence. The system provides soil classification, pest and disease detection using YOLOv8, and a multilingual chatbot for real-time agricultural guidance.

This project focuses on making modern agricultural technology simple, accessible, and useful for farmers through an easy-to-use digital interface.

------------------------------------------------------------

## PROJECT OVERVIEW

AgriSmart helps farmers by:
- Identifying soil type using image-based classification
- Detecting crop pests and diseases at early stages
- Providing crop recommendations
- Offering agricultural guidance through a multilingual chatbot

The platform bridges the gap between traditional farming practices and modern AI-driven solutions.

------------------------------------------------------------

## FEATURES

- Soil type prediction using CNN-based image classification
- Pest and disease detection using YOLOv8
- Multilingual chatbot support (English, Tamil, Hindi, Telugu, Malayalam, Kannada)
- Crop recommendations based on soil type
- User login and profile management
- Fast prediction response time
- Simple and user-friendly interface for rural users

------------------------------------------------------------

## PROBLEM STATEMENT

Farmers often face challenges such as:
- Lack of awareness about soil type
- Delay in identifying plant diseases
- Limited access to expert agricultural guidance
- Language barriers in using digital tools

AgriSmart solves these problems using AI models and web technologies to provide accurate, fast, and accessible farming support.

------------------------------------------------------------

## TECHNOLOGY STACK

# Frontend:
- HTML
- CSS
- JavaScript

# Backend:
- Python (Flask)

# AI/ML Models:
- Convolutional Neural Network (CNN) for soil classification
- YOLOv8 for pest and disease detection

# Database:
- SQLite

# APIs:
- Gemini API for chatbot functionality
- Translation APIs for multilingual communication

## Development Tools:
- Visual Studio Code
- Git and GitHub
- OpenCV
- TensorFlow / Keras
- Ultralytics YOLOv8

------------------------------------------------------------

## PROJECT MODULES

1. Soil Classification Module
- Upload soil image
- CNN predicts soil type
- System suggests suitable crops

2. Pest and Disease Detection Module
- Upload crop leaf image
- YOLOv8 detects pest or disease
- Displays treatment suggestions

3. Chatbot Module
- Accepts queries in multiple languages
- Provides farming guidance and recommendations

4. User Module
- User registration and login
- Stores prediction history and interactions

------------------------------------------------------------

## HOW IT WORKS

1. User logs into the system.
2. Uploads a soil or crop image.
3. The AI model processes the image.
4. The system displays prediction results instantly.
5. The chatbot can be used for additional farming queries.

------------------------------------------------------------

MODEL PERFORMANCE

- Soil Classification Accuracy: Approximately 94%
- Pest Detection Accuracy (YOLOv8): Approximately 92%
- Chatbot Query Response Success Rate: Approximately 95%
- Average Response Time: Less than 2 seconds

------------------------------------------------------------

## INSTALLATION AND SETUP

1. Clone the repository

git clone https://github.com/Areya911/AgriSmart-Mini_proj.git
cd AgriSmart-Mini_proj

2. Create a virtual environment

python -m venv venv
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Run the application

python app.py

5. Open in browser

http://127.0.0.1:5000/

------------------------------------------------------------
## IMPACT

AgriSmart helps in:
- Improving crop yield through informed decisions
- Detecting plant diseases early
- Reducing dependency on manual soil testing
- Making AI accessible to rural farmers
- Supporting sustainable farming practices

------------------------------------------------------------

## FUTURE ENHANCEMENTS

- Mobile application for easier access
- Weather-based crop recommendation
- Larger datasets for improved model accuracy
- Voice-based assistant support
- Advanced analytics dashboard

