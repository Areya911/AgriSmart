# 🌱 AgriSmart — AI-Powered Smart Farming Assistant

> Empowering farmers with real-time AI insights for soil health, crop disease detection, and smart agricultural guidance.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-green?logo=flask)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)](https://tensorflow.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://ultralytics.com)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46e3b7?logo=render)](https://render.com)

---

## 🚀 Live Demo

| Resource | Link |
|---|---|
| **Live Application** | https://your-live-url.onrender.com |
| **GitHub Repository** | https://github.com/Areya911/AgriSmart |

---

## 📖 Project Overview

AgriSmart is a full-stack AI-powered web application designed to support farmers across India with intelligent, multilingual agricultural assistance. The platform integrates computer vision, deep learning, and generative AI to provide:

- **Soil type classification** from uploaded or camera-captured images
- **Plant disease detection** with treatment recommendations
- **Voice-enabled AI chatbot** (Whisper ASR + Gemini + gTTS)
- **Multilingual support** (English, Hindi, Tamil, Telugu, Kannada, Malayalam)
- **Personalized profiles** with farming history and crop recommendations

---

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgriSmart                                │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │  Flask   │   │  TensorFlow  │   │   Google Gemini AI   │    │
│  │  App     │──▶│  CNN Models  │   │   (Chatbot & Advice) │    │
│  │ (Python) │   │  Soil + Leaf │   └──────────────────────┘    │
│  └──────────┘   └──────────────┘                               │
│       │                                                         │
│       ├──▶  ┌──────────────┐   ┌──────────────────────────┐   │
│       │     │  YOLOv8      │   │  OpenAI Whisper (ASR)    │   │
│       │     │  Object Det. │   │  gTTS (Text-to-Speech)   │   │
│       │     └──────────────┘   └──────────────────────────┘   │
│       │                                                         │
│       └──▶  ┌────────────────────────────────────────────┐    │
│             │  SQLite DB  │  Flask-WTF CSRF  │  Limiter   │    │
│             └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

Model Storage (Google Drive) ──▶ Auto-downloaded at first startup
                                 Cached locally for subsequent runs
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10, Flask 3.1, Gunicorn |
| **Database** | SQLite (SQLAlchemy ORM) |
| **Soil Classification** | Keras/TensorFlow CNN (custom trained) |
| **Plant Disease Detection** | MobileNetV2 fine-tuned + YOLOv8 nano |
| **Speech-to-Text** | OpenAI Whisper (base model) |
| **Text-to-Speech** | gTTS (Google Text-to-Speech) |
| **AI Chatbot** | Google Gemini 1.5 Flash |
| **Security** | Flask-WTF CSRF, Flask-Limiter, bleach |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript (ES6+) |
| **Deployment** | Render (Web Service) |

---

## ✨ Features

| Feature | Description |
|---|---|
| 🌍 **Soil Classifier** | Upload or capture soil image → predict type → recommend crops |
| 🌿 **Plant Disease Detector** | YOLO + MobileNetV2 dual-stage disease analysis with remedy tips |
| 🤖 **AgriBot Chatbot** | Voice or text chat powered by Gemini with farming context |
| 🎙️ **Voice Input (ASR)** | Record voice messages → Whisper transcription → AI response |
| 🌐 **6 Languages** | English, हिन्दी, தமிழ், తెలుగు, ಕನ್ನಡ, മലയാളം |
| 👤 **User Profiles** | Crop history, soil results, region-based recommendations |
| 🛡️ **Admin Dashboard** | View all registered users and their activity |
| ☁️ **Weather Widget** | Live weather data via OpenWeatherMap for user's district |

---

## 📦 Local Setup

### Prerequisites
- Python 3.10+
- `ffmpeg` installed and on PATH (required for Whisper audio processing)
- Git

### 1. Clone the repository
```bash
git clone https://github.com/Areya911/AgriSmart.git
cd AgriSmart
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env and fill in your API keys and Google Drive model IDs
```

### 5. Upload model files to Google Drive and get File IDs
The trained model weights are **not in this repository** (too large for GitHub).
You must upload them to Google Drive and add the file IDs to your `.env`:

| Model File | `.env` Key |
|---|---|
| `soil_classifier.h5` | `SOIL_MODEL_GDRIVE_ID` |
| `leaf_disease_mobilenet_finetuned.h5` | `LEAF_MODEL_GDRIVE_ID` |
| `yolov8n.pt` | `YOLO_MODEL_GDRIVE_ID` |

**How to get a Google Drive File ID:**
1. Upload the file to Google Drive
2. Right-click → **Share** → **Anyone with the link** → Copy
3. The ID is the long string in the URL:
   `https://drive.google.com/file/d/`**`<THIS_IS_THE_FILE_ID>`**`/view`

### 6. Run the application
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

> **Note:** On first run, models are automatically downloaded from Google Drive (takes ~1–2 minutes depending on connection). Subsequent runs use the cached files.

---

## 🚀 Deployment on Render

### Step 1 — Push to GitHub
```bash
git add .
git commit -m "Production-ready deploy"
git push origin main
```

### Step 2 — Create a Render Web Service
1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New +** → **Web Service**
3. Connect your GitHub account and select **Areya911/AgriSmart**

### Step 3 — Configure Service Settings
| Setting | Value |
|---|---|
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT wsgi:application` |
| **Instance Type** | Free |

### Step 4 — Set Environment Variables
In the **Environment** tab, add all variables from `.env.example`:

| Key | Value |
|---|---|
| `PYTHON_VERSION` | `3.10.14` |
| `FLASK_ENV` | `production` |
| `FLASK_SECRET` | *(generate a long random string)* |
| `GEMINI_API_KEY` | *your Google Gemini API key* |
| `OPENWEATHER_API_KEY` | *your OpenWeatherMap API key* |
| `ADMIN_USERNAME` | *your admin email* |
| `ADMIN_PASSWORD` | *your admin password* |
| `SOIL_MODEL_GDRIVE_ID` | *Google Drive file ID for soil_classifier.h5* |
| `LEAF_MODEL_GDRIVE_ID` | *Google Drive file ID for leaf_disease_mobilenet_finetuned.h5* |
| `YOLO_MODEL_GDRIVE_ID` | *Google Drive file ID for yolov8n.pt* |

### Step 5 — Deploy!
Click **Create Web Service**. Render will:
1. Install Python 3.10.14
2. Run `pip install -r requirements.txt`
3. Start Gunicorn
4. On first request: auto-download models from Google Drive → load into memory

> ⚠️ **Free tier cold starts**: Render free services spin down after 15 min of inactivity. The first request after spin-down may take 60–90 seconds (model download + load). Subsequent requests are fast.

---

## 📋 Deployment Checklist

- [ ] Python 3.10+ installed locally
- [ ] `ffmpeg` installed on system PATH
- [ ] All packages installed: `pip install -r requirements.txt`
- [ ] `.env` file created with all keys filled in
- [ ] Model files uploaded to Google Drive (public sharing enabled)
- [ ] Google Drive File IDs added to `.env`
- [ ] App tested locally: `python app.py` → all routes return 200 OK
- [ ] Code pushed to GitHub: `git push origin main`
- [ ] Render Web Service created and connected to GitHub repo
- [ ] All environment variables set in Render dashboard
- [ ] Render build completes without errors
- [ ] Live URL opens and all features work end-to-end

---

## 🔒 Security Features

- **CSRF Protection**: Flask-WTF tokens on all forms and AJAX requests
- **Rate Limiting**: Flask-Limiter on `/login`, `/signup`, `/contact`
- **File Upload Hardening**: Extension allowlist + PIL re-encoding of profile images
- **Output Sanitization**: `bleach` on all AI-generated content
- **Security Headers**: CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy
- **Secure Sessions**: HttpOnly + SameSite=Lax cookie flags
- **No Hardcoded Secrets**: All credentials loaded from environment variables

---

## 📁 Repository Structure

```
AgriSmart/
├── app.py                    # Main Flask application
├── models.py                 # SQLAlchemy database models
├── speech_utils.py           # Whisper ASR + gTTS utilities
├── model_downloader.py       # Auto-downloads models from Google Drive
├── wsgi.py                   # Gunicorn WSGI entry point
├── requirements.txt          # Python dependencies
├── Procfile                  # Process file for Render/Heroku
├── render.yaml               # Render deployment blueprint
├── runtime.txt               # Python version pin
├── .env.example              # Environment variable template
├── .gitignore                # Excludes models, datasets, secrets
├── robots.txt                # Search crawler configuration
├── sitemap.xml               # SEO sitemap
├── translations/             # i18n JSON files (6 languages)
├── templates/                # Jinja2 HTML templates
├── static/
│   ├── js/                   # Frontend JavaScript
│   ├── css/                  # Stylesheets
│   ├── crop_images/          # Crop flipcard images
│   └── icons/                # SVG icon assets
└── README.md
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License.
