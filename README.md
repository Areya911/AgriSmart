# AgriSmart — Smart Guidance for Crops & Soil

AgriSmart is an AI-powered smart agricultural assistant designed to empower farmers with real-time, multilingual, and secure farming insights. It leverages Deep Learning models for soil classification and crop recommendation, computer vision (YOLOv8 & MobileNetV2) for plant disease analysis, and Generative AI (Gemini) for smart conversational support via text and voice.

---

## 🌟 Key Features

1. **Soil Classifier**: Upload a soil image or capture it live to predict the soil type (Alluvial, Black, Clay, Red, Sandy, etc.) and get instant crop recommendations.
2. **Plant Health & Disease Analyzer**: Dual-stage analysis using:
   - **YOLOv8** to locate and crop leaves.
   - **MobileNetV2** (Fine-tuned) to classify specific leaf diseases and recommend chemical and organic remedies.
3. **AgriBot AI Chatbot**: Conversational assistant powered by Gemini. Supports voice speech-to-text (Whisper ASR) and text-to-speech (gTTS) feedback with automatic language detection.
4. **Multilingual Interface**: Fully translated pages in English, हिन्दी (Hindi), தமிழ் (Tamil), తెలుగు (Telugu), ಕನ್ನಡ (Kannada), and മലയാളം (Malayalam).
5. **Admin Portal**: Restricted administrative dashboard to monitor registered users.

---

## 🔒 Security Hardening (OWASP Compliant)

This application has been comprehensively audited and hardened against the OWASP Top 10 vulnerabilities:

* **Authentication (OWASP A07:2021)**: Removed all hardcoded administrator credentials. Admin credentials are dynamically loaded from environment variables (`ADMIN_USERNAME` & `ADMIN_PASSWORD`).
* **CSRF Protection (OWASP A01:2021)**: Global Cross-Site Request Forgery (CSRF) protection enabled via `Flask-WTF`. All forms and AJAX POST actions validate anti-forgery tokens.
* **Unrestricted File Upload Hardening (OWASP A03:2021)**:
  - Globals upload sizes capped at 16MB (`MAX_CONTENT_LENGTH`).
  - Image uploads restricted to an allowed extension whitelist (`png`, `jpg`, `jpeg`, `gif`, `webp`).
  - Uploaded profile photos are parsed and re-encoded using `Pillow (PIL)` to strip metadata and prevent steganographic execution payloads.
  - Removed absolute filesystem path disclosure in API responses.
* **Output Sanitization (OWASP A03:2021)**: Incorporated `bleach` to sanitize and escape AI model outputs (`ai_advice`) before context rendering to eliminate Stored/Reflected XSS.
* **Security HTTP Headers (OWASP A05:2021)**: Injects security headers on all responses:
  - `X-Frame-Options: DENY` (prevents Clickjacking)
  - `X-Content-Type-Options: nosniff` (prevents MIME sniffing)
  - `Referrer-Policy: strict-origin-when-cross-origin`
  - `Content-Security-Policy (CSP)`
* **Session Hardening**: Sessions configured with `SESSION_COOKIE_HTTPONLY=True`, `SESSION_COOKIE_SAMESITE='Lax'`, and conditional `SESSION_COOKIE_SECURE=True`.
* **Rate Limiting (OWASP A04:2021)**: Integrated `Flask-Limiter` to protect `/login`, `/signup`, and `/contact` routes against brute-force attacks and abuse.

---

## 🛠️ Technology Stack

- **Backend**: Python, Flask, Flask-SQLAlchemy (SQLite)
- **Frontend**: Semantic HTML5, Vanilla CSS3 (Green branding theme), JavaScript (ES6+)
- **Machine Learning**: TensorFlow (MobileNetV2), PyTorch (YOLOv8), OpenAI Whisper, Google Gemini API
- **Deployment**: Gunicorn (WSGI), Flask-Compress (Gzip compression)

---

## ⚙️ Installation & Local Setup

### Prerequisites
- Python 3.10 or higher
- `ffmpeg` installed on the system path (required for audio transcription)

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Areya911/AgriSmart.git
   cd AgriSmart
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables**:
   Create a `.env` file in the root folder:
   ```env
   GEMINI_API_KEY=your_gemini_key
   OPENWEATHER_API_KEY=your_openweather_key
   FLASK_SECRET=your_secure_session_secret
   ADMIN_USERNAME=admin@org.in
   ADMIN_PASSWORD=your_secure_admin_password
   ```

5. **Run the Application**:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your web browser.

---

## 🚀 Hosting & Deployment Recommendations

### ⚠️ Note on Vercel
While Vercel is highly recommended for static frontends and serverless Jamstack architectures, **Vercel is not suitable for this application** due to the following structural limitations:
1. **Deployment Size Limit**: Vercel enforces a serverless function payload limit of **50MB**. The dependencies for this application (TensorFlow, PyTorch, YOLO, OpenAI Whisper, OpenCV) exceed **1.5GB**, making serverless packaging impossible.
2. **Cold Start Timeouts**: Loading the pre-trained Keras models (`.h5`) and PyTorch weights into memory during a cold start exceeds Vercel's serverless execution timeout limits (10-15 seconds on free/hobby plans).

### Recommended Staging/Production Platforms

#### 1. Render (Recommended Web Service)
Render is an excellent platform for hosting machine learning Flask applications:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --workers 1 --threads 4 --bind 0.0.0.0:$PORT wsgi:application`
- **Environment Settings**: Add environment variables directly in Render's Env Settings panel.

#### 2. Hugging Face Spaces (Docker / Streamlit / Gradio / Flask)
You can deploy the app as a Hugging Face Space using their Docker runtime. It provides a free container with 16GB RAM, which is ideal for running TensorFlow and PyTorch models without memory issues.
- Create a `Dockerfile` for the application.
- Expose port `7860`.
