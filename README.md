AgriSmart – AI Powered Smart Farming Assistant:
AgriSmart is a web-based smart agriculture platform that helps farmers make better decisions using Artificial Intelligence.
The system provides soil classification, pest and disease detection using YOLOv8, and a multilingual chatbot for instant agricultural guidance.
This project focuses on making AI simple, accessible, and useful for farmers through an easy-to-use interface.

Features:
-Soil type prediction using image classification (CNN)
-Pest & disease detection using YOLOv8
-Multilingual AI chatbot (English, Tamil, Hindi, Telugu, Malayalam, Kannada)
-Crop recommendations based on soil type
-User login & profile management
-AI-powered insights for better farming decisions
-Fast response time (< 2 seconds)

Problem Statement:
Farmers often face challenges like:
1. Not knowing soil type accurately
2. Late detection of crop diseases
3. Limited access to expert guidance
4. Language barriers in using digital tools
5. AgriSmart solves these problems using AI + Web Technology.

🏗️ Tech Stack
Frontend=> HTML, CSS, JavaScript
Backend=> Python (Flask), AI/ML Models, CNN – Soil Classification, YOLOv8 – Pest & Disease Detection
Database=> SQLite, APIs, Gemini API (Chatbot)
Translation APIs (Multilingual support)

Tools:
VS Code
Git & GitHub
OpenCV
TensorFlow / Keras
Ultralytics YOLOv8

Project Modules
1️⃣ Soil Classification Module
Upload soil image
CNN predicts soil type
Suggests suitable crops

2️⃣ Pest & Disease Detection Module
Upload crop leaf image
YOLOv8 detects disease
Suggests treatment methods

3️⃣ Chatbot Module
Ask questions in regional languages
Get instant farming advice

4️⃣ User Module
Login / Signup
View prediction history

How It Works:
->User logs into the system
->Uploads soil or crop image
AI model processes image
Results shown instantly
Chatbot available for additional help

Model Performance
Soil Classification Accuracy: ~94%
Pest Detection Accuracy (YOLOv8): ~92%
Chatbot Query Success Rate: ~95%

Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/Areya911/AgriSmart-Mini_proj.git
cd AgriSmart-Mini_proj

2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the application
python app.py

5️⃣ Open in browser
http://127.0.0.1:5000/


Impact=>
AgriSmart helps:
-Improve crop yield
-Detect diseases early
-Reduce farming losses
-Make AI accessible to rural farmers
-Promote smart & sustainable agriculture

Future Improvements:
1. Mobile app version
2. Weather-based crop recommendation
3. Larger dataset training
4. Voice assistant support
5. IoT integration for real-time soil data
