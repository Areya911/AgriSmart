# app.py (updated)
import os
import json
from pathlib import Path
import time
import requests
from werkzeug.security import generate_password_hash, check_password_hash
import tempfile
import subprocess
import re
from flask import send_from_directory, current_app, Response
import base64
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, g
from PIL import Image,ImageDraw, ImageFont
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import google.generativeai as genai
import shutil
from dotenv import load_dotenv
from collections import Counter
import html
from models import db, User, ChatHistory


def clean_markdown(text: str) -> str:
    """
    Aggressively strip common Markdown from model replies.
    Keeps the readable text, removes bullets, bold/italic markers, inline code ticks,
    and collapses excessive blank lines. Returns plain text.
    """
    if not text:
        return ""
    try:
        t = str(text)

        # Normalize newlines
        t = t.replace('\r\n', '\n').replace('\r', '\n')

        # Remove fenced code blocks ```...```
        t = re.sub(r"```[\s\S]*?```", "", t, flags=re.M)

        # Remove headings like ### or ## at start of line
        t = re.sub(r"^\s{0,3}#{1,6}\s+", "", t, flags=re.M)

        # Remove list markers at line start (allow leading spaces/tabs)
        t = re.sub(r"^[ \t]*[\*\-\+]\s+", "", t, flags=re.M)

        # Remove ordered list numbers "1. " at line start
        t = re.sub(r"^[ \t]*\d+\.\s+", "", t, flags=re.M)

        # Remove bold/italic markers while keeping the content
        t = re.sub(r"\*\*(.*?)\*\*", r"\1", t, flags=re.S)
        t = re.sub(r"__(.*?)__", r"\1", t, flags=re.S)
        t = re.sub(r"\*(.*?)\*", r"\1", t, flags=re.S)
        t = re.sub(r"_(.*?)_", r"\1", t, flags=re.S)

        # Remove inline code markers `like this`
        t = re.sub(r"`(.*?)`", r"\1", t, flags=re.S)

        # Remove any remaining stray asterisks (1-3)
        t = re.sub(r"\*{1,3}", "", t)

        # Trim trailing space on lines
        t = "\n".join(line.rstrip() for line in t.splitlines())

        # Collapse 3+ blank lines to two
        t = re.sub(r"\n{3,}", "\n\n", t)

        return t.strip()
    except Exception:
        return str(text)

# ensure static tts folder exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TTS_DIR = os.path.join(BASE_DIR, "static", "tts")
os.makedirs(TTS_DIR, exist_ok=True)


YOLO_OUT_DIR = os.path.join(BASE_DIR, "static", "yolo_out")
os.makedirs(YOLO_OUT_DIR, exist_ok=True)
import json
import glob
from flask import g

# Simple i18n: load translations from translations/*.json (filenames are language codes)
TRANSLATIONS = {}

def load_all_translations():
    """Load all translations from translations/*.json into TRANSLATIONS.
    This function is safe to call before app exists (uses prints for errors)."""
    global TRANSLATIONS
    TRANSLATIONS = {}
    trans_dir = os.path.join(BASE_DIR, "translations")
    if not os.path.isdir(trans_dir):
        # no translations yet — that's fine
        return
    for path in glob.glob(os.path.join(trans_dir, "*.json")):
        code = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                TRANSLATIONS[code] = { str(k): str(v) for k, v in data.items() }
        except Exception as exc:
            # avoid using current_app here (no app context). Print helpful message.
            print(f" Failed to load translation file {path}: {exc}")

# load at startup (call this after BASE_DIR is defined)



def analyze_image_array(img_array, conf_thresh=0.20, imgsz=640):
    """
    Run YOLO directly on a numpy array (no file).
    Returns (detected_label, status, advice, annotation_url)
    """
    if yolo_model is None:
        return "Unknown", "General", "No YOLO annotation available (model missing).", None

    try:
        results = yolo_model.predict(img_array, conf=conf_thresh, imgsz=imgsz)
        if not results or len(results) == 0:
            return "Unknown", "General", "No YOLO results returned.", None

        r = results[0]
        names = getattr(r, "names", {}) or {}
        boxes = getattr(r, "boxes", None)

        if boxes is None or len(boxes) == 0:
            return "Unknown", "General", "No detections (boxes empty).", None

        # collect info
        cls_indices = [int(x) for x in boxes.cls]
        confs = [float(x) for x in boxes.conf]
        xyxy = [list(map(float, b)) for b in boxes.xyxy]

        # top detection
        top_idx = int(max(range(len(confs)), key=lambda i: confs[i]))
        top_cls = cls_indices[top_idx]
        top_label = names[top_cls] if isinstance(names, dict) and top_cls in names else str(top_cls)
        top_conf = confs[top_idx]

        # health status heuristic
        status = "General"
        if "disease" in top_label.lower() or top_label.lower().startswith(("blight", "rust")):
            status = "Diseased"
        elif top_label != "Unknown":
            status = "Healthy"

        # AI advice if diseased
        advice = ""
        if status == "Diseased" and gemini_model is not None:
            prompt = f"The image shows a {top_label} with a disease. Give cause, disease name, and remedies (organic + chemical)."
            try:
                resp = gemini_model.generate_content(prompt)
                advice = getattr(resp, "text", str(resp))
            except Exception:
                advice = "Gemini advice not available."

        # annotate image
        try:
            pil = Image.fromarray(img_array.astype('uint8'), 'RGB')
            draw = ImageDraw.Draw(pil)
            try:
                font = ImageFont.truetype("arial.ttf", size=18)
            except Exception:
                font = ImageFont.load_default()

            for xy, cls_i, conf in zip(xyxy, cls_indices, confs):
                label = names.get(cls_i, str(cls_i)) if isinstance(names, dict) else str(cls_i)
                x1, y1, x2, y2 = map(int, xy)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1+4, y1+4), f"{label} {conf:.2f}", fill="white", font=font)

            out_name = f"yolo_{int(time.time()*1000)}.jpg"
            out_path = os.path.join(YOLO_OUT_DIR, out_name)
            pil.save(out_path, quality=85)
            annotation_url = url_for('static', filename=f"yolo_out/{out_name}", _external=False)
        except Exception:
            annotation_url = None

        return top_label, status, advice, annotation_url

    except Exception as e:
        app.logger.exception("YOLO analyze error")
        return "Unknown", "General", f"YOLO Error: {str(e)}", None

#-------------------------------------------------------
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # set in .env

def geocode_location(state, district, country="India"):
    """
    Use OpenWeatherMap geocoding API to get lat/lon for a location string.
    Returns (lat, lon) or (None, None).
    """
    if not OPENWEATHER_API_KEY:
        return None, None
    q = f"{district}, {state}, {country}"
    url = "http://api.openweathermap.org/geo/1.0/direct"
    try:
        r = requests.get(url, params={"q": q, "limit": 1, "appid": OPENWEATHER_API_KEY}, timeout=8)
        r.raise_for_status()
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        current_app.logger.exception("Geocode failed")
    return None, None

def get_current_weather_by_latlon(lat, lon):
    """
    Query OpenWeatherMap current weather. Returns a dict with key info or None.
    """
    if not OPENWEATHER_API_KEY or not (lat and lon):
        return None
    url = "https://api.openweathermap.org/data/2.5/weather"
    try:
        r = requests.get(url, params={"lat": lat, "lon": lon, "units": "metric", "appid": OPENWEATHER_API_KEY}, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        current_app.logger.exception("Weather fetch failed")
    return None

# Helper: fetch weather for the logged-in user (from profile)
def get_weather_for_user(username):
    """
    Use SQLAlchemy User model to fetch state/district and then call geocoding/weather.
    Returns weather dict or None.
    """
    try:
        user = User.query.filter_by(username=username).first()
        if not user or not user.state or not user.district:
            return None
        lat, lon = geocode_location(user.state, user.district)
        if lat is None:
            return None
        return get_current_weather_by_latlon(lat, lon)
    except Exception:
        current_app.logger.exception("get_weather_for_user failed")
        return None

# ----------------- Load env -----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FLASK_SECRET = os.getenv("FLASK_SECRET", "dev-secret")
api_key = os.getenv("OPENWEATHER_API_KEY")
print("My Weather API Key:", api_key)
# ----------------- Flask app -----------------
app = Flask(__name__)
app.secret_key = FLASK_SECRET

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agri.db'  # or your DB
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()

# ----------------- Multilingual support (single canonical block) -----------------
import glob
from flask import g

# Ensure translations were loaded at program start
load_all_translations()

@app.route('/set_language', methods=['POST'])
def set_language():
    """
    Save user's language selection to session.
    If the user is logged in you may also persist this preference to DB (optional).
    The form should include a "next" hidden field to redirect back.
    """
    try:
        lang = (request.form.get('language') or 'en').strip().lower()
        # basic validation — accept only known codes
        allowed = ('en', 'ta', 'kn', 'ml', 'te', 'hi')
        if lang not in allowed:
            lang = 'en'
        session['language'] = lang

        # Optional: if you want to persist for logged-in users, uncomment the block below
        # and implement a proper user preferences storage (DB or file).
        # if 'username' in session:
        #     # Example: add a user_prefs table or update your user profile in DB here.
        #     try:
        #         # pseudo-code: save_user_pref(session['username'], 'language', lang)
        #         pass
        #     except Exception:
        #         current_app.logger.exception("Failed to persist user language preference")

        # redirect back to the page the user was on (or to home)
        next_url = request.form.get('next') or url_for('home')
        return redirect(next_url)
    except Exception as e:
        current_app.logger.exception("set_language error")
        # fallback redirect
        return redirect(url_for('home'))

# ensure session language exists default on each request
@app.before_request
def set_locale():
    # prefer explicit session value; fallback to request accept languages or 'en'
    lang = session.get('language')
    if not lang:
        # try to match browser pref
        try:
            best = request.accept_languages.best_match(list(TRANSLATIONS.keys()) or ['en'])
            lang = best or 'en'
        except Exception:
            lang = 'en'
        session.setdefault('language', lang)
    g.lang = lang
    g.trans = TRANSLATIONS.get(lang, TRANSLATIONS.get('en', {}))

def translate(key: str, default: str = None) -> str:
    """
    Lookup translation for 'key' in current g.trans, fallback to english then default.
    Use in templates as tr('KEY_NAME') or in server code via translate('KEY').
    """
    default = default if default is not None else key
    # prefer current lang
    val = None
    try:
        if getattr(g, 'trans', None):
            val = g.trans.get(key)
    except Exception:
        val = None
    # fallback to english
    if not val:
        val = TRANSLATIONS.get('en', {}).get(key)
    return val if val is not None else default

# register in jinja so templates can call tr("KEY")
@app.context_processor
def inject_translate():
    return {
        'tr': translate,
        'current_lang': lambda: getattr(g, 'lang', session.get('language', 'en'))
    }

# --- Paste into app.py (after your existing imports) ---
import os
import time
import tempfile
import subprocess
from flask import send_from_directory, current_app

# optional imports - whisper for local transcription, gTTS for TTS
try:
    import whisper
    WHISPER_AVAILABLE = True
    # load model on first request lazily - we will initialize later to avoid startup hit
    _whisper_model = None
except Exception:
    WHISPER_AVAILABLE = False
    _whisper_model = None

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# ensure static tts folder exists

def ensure_whisper_model(model_name="medium"):
    """
    Lazily load whisper model to avoid blocking app startup.
    """
    global _whisper_model
    if not WHISPER_AVAILABLE:
        return None
    if _whisper_model is None:
        _whisper_model = whisper.load_model(model_name)
    return _whisper_model

def webm_to_wav(webm_path, wav_path, ffmpeg_exec=None):
    ff = ffmpeg_exec or globals().get('FFMPEG_EXEC') or resolve_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg executable not found.")
    # Normalize, remove silence, denoise, and downsample 16k mono
    cmd = [
        ff, '-y', '-i', webm_path,
        '-af', 'silenceremove=1:0:-50dB,highpass=f=80,lowpass=f=8000,afftdn,loudnorm',
        '-ar', '16000', '-ac', '1',
        wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def transcribe_with_whisper(wav_path, model_name="small", language_hint=None):
    model = ensure_whisper_model(model_name)
    if model is None:
        raise RuntimeError("Whisper model not available.")

    # Farming keywords to help Whisper bias recognition
    farming_prompt = (
        "agriculture, farming, soil, crop, irrigation, fertilizer, pesticide, "
        "wheat, paddy, rice, maize, millet, groundnut, cotton, sugarcane, "
        "disease, blight, rust, sow, alluvial, black, red, clay ,leaf spot, pest control, organic, yield, harvest"
    )

    # Options for Whisper
    options = {
        "language": language_hint or None,   # auto-detect if None
        "task": "transcribe",
        "initial_prompt": farming_prompt     # 👈 boost agri vocabulary
    }

    result = model.transcribe(wav_path, **options)
    text = result.get("text", "").strip()
    lang = result.get("language", None) or "en"
    return text, lang



def query_gemini_reply(user_text, lang_code=None):
    """
    Multilingual reply helper for Gemini.
    user_text: the transcribed speech text
    lang_code: ISO code like 'en','hi','ta','te','ml','kn'
    Returns: plain text reply in that language
    """
    hint = (lang_code or "en")[:2]
    lang_names = {
        "en": "English", "hi": "Hindi", "ta": "Tamil",
        "te": "Telugu", "ml": "Malayalam", "kn": "Kannada"
    }
    lang_name = lang_names.get(hint, hint)

    system_prompt = (
        "IMPORTANT: Reply exclusively in the user's language. "
        "Do not switch to English unless the user spoke English.\n\n"
        f"You are AgriSmart, an agricultural assistant. "
        f"The user is speaking {lang_name} (code {hint}). "
        "Be concise and relevant to farming."
    )

    try:
        resp = gemini_model.generate_content(
            f"{system_prompt}\n\nUser: {user_text}"
        )
        reply_text = getattr(resp, "text", None) or str(resp)
        return reply_text.strip()
    except Exception as e:
        current_app.logger.exception("Gemini reply failed")
        return "Sorry — I couldn't generate a response right now."


# helper to call your LLM (Gemini via google.generativeai)
# expose helper so blueprints can call it without import cycles
# add this after your query_gemini_reply(...) definition in app.py
app.query_gemini_reply = query_gemini_reply


# Flask route for ASR
from flask import request, jsonify


# Resolve ffmpeg safely once app exists
def resolve_ffmpeg():
    ff = shutil.which("ffmpeg")
    if ff and os.path.isfile(ff):
        return ff

    # fallback path (Windows example) - adjust or remove if not needed
    fallback = r"C:\Users\USER\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
    if os.path.isfile(fallback):
        app.logger.info(f"Using fallback ffmpeg at: {fallback}")
        return fallback

    app.logger.warning("FFmpeg not found on PATH. ASR endpoint will return helpful error.")
    return None

FFMPEG_EXEC = resolve_ffmpeg()

# Register voice blueprint AFTER FFMPEG_EXEC is resolved and after app is created
# (we removed the earlier import to avoid import-time cycles)
from voice_blueprint import voice_bp
import voice_blueprint as _voice_bp_mod

# pass the resolved ffmpeg path into the blueprint module
_voice_bp_mod.FFMPEG_EXEC = FFMPEG_EXEC

# ensure app exposes the Gemini helper (you already set this earlier)
# now register the blueprint
app.register_blueprint(voice_bp)


# ----------------- Configure Gemini -----------------
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in .env. Chatbot will fail until key is configured.")
genai.configure(api_key=GEMINI_API_KEY)

# Create a Gemini model object
# Use a stable model name you have access to
try:
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    # optional quick smoke test (comment out if noisy)
    # test_response = gemini_model.generate_content("Say hi in one sentence.")
    # print("Gemini Test (first words):", test_response.text[:120])
except Exception as e:
    gemini_model = None
    print("Could not initialize gemini_model:", str(e))

# ----------------- Load ML models -----------------
# Load your trained Keras model and YOLO weights (if present).
# If models are missing, start app but gracefully handle inference errors.
soil_model = None
yolo_model = None
try:
    soil_model = load_model("soil_classifier.h5")
    print("Loaded soil_classifier.h5")
except Exception as e:
    print("Could not load soil_classifier.h5:", e)

try:
    yolo_model = YOLO("yolov8n.pt")
    print("Loaded YOLOv8 model")
except Exception as e:
    print("Could not load yolov8n.pt:", e)

@app.route("/stream_chat", methods=["GET"])
def stream_chat():
    if "username" not in session:
        return Response("not_logged_in", status=401)

    user_query = request.args.get("q", "").strip()
    if not user_query:
        return Response("empty_query", status=400)

    def generate():
        try:
            for chunk in gemini_model.generate_content(user_query, stream=True):
                piece = getattr(chunk, "text", None)
                if piece:
                    yield f"data: {piece}\n\n"
        except Exception as e:
            yield f"data: [Error: {str(e)}]\n\n"

    return Response(generate(), mimetype="text/event-stream")
# ----------------- Soil class mapping (optional) -----------------
# If you produced class_map.json at training time, you can load it here.
# Fallback to the basic mapping if not present (but real mapping should come from training script).
soil_classes = ["alluvial", "black", "clay", "red"]
soil_crop_dict = { "alluvial": ["Wheat", "Paddy", "Sugarcane", "Jute"], "black": ["Cotton", "Soybean", "Jowar"], "clay": ["Rice", "Sugarcane"], "peat": ["Rice", "Pineapple"], "red": ["Groundnut", "Millet", "Ragi"], "sandy": ["Carrot", "Potato", "Peanut"], "yellow": ["Pulses", "Oilseeds"] }

# ----------------- Database (SQLite) for chat history -----------------



# AJAX endpoint to save chat from client (used by voice flow)
@app.route('/save_chat_ajax', methods=['POST'])
def save_chat_ajax():
    data = request.get_json()
    user_query = data.get("user_query") or data.get("user_text", "")

    if not user_query:
        return jsonify(ok=False, error="Empty query")

    # Use central reply generator
    bot_response = generate_bot_reply(user_query, session.get("username"))

    # Save chat correctly
    chat = ChatHistory(
        username=session.get("username"),
        user_message=user_query,
        bot_response=bot_response,
        timestamp=datetime.now()
    )
    db.session.add(chat)
    db.session.commit()

    return jsonify(ok=True, user_text=user_query, bot_response=bot_response)

def generate_bot_reply(user_text, username=None, lang_code="en"):
    """
    Centralized bot reply generator.
    Uses query_gemini_reply + markdown cleaner.
    """
    try:
        raw = query_gemini_reply(user_text, lang_code=lang_code)
        return clean_markdown(raw)
    except Exception:
        current_app.logger.exception("generate_bot_reply failed")
        return f"Bot response for: {user_text}"


@app.route("/ajax_chat", methods=["POST"])
def ajax_chat():
    if "username" not in session:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401

    data = request.get_json(silent=True) or {}
    user_query = (data.get("user_query") or "").strip()
    if not user_query:
        return jsonify({"ok": False, "error": "empty_query"}), 400
    
    lang_code = session.get("language") or "en"
    bot_text = generate_bot_reply(user_query, session.get("username"), lang_code=lang_code)

    # ✅ Add TTS here
    tts_url = None
    try:
        if GTTS_AVAILABLE and bot_text:
            from gtts import gTTS
            tts = gTTS(bot_text, lang=(lang_code or "en"))
            filename = f"tts_{int(time.time()*1000)}.mp3"
            path = os.path.join(TTS_DIR, filename)
            tts.save(path)
            tts_url = url_for("static", filename=f"tts/{filename}", _external=False)
    except Exception:
        current_app.logger.exception("TTS generation failed in /ajax_chat")

    # Save chat in DB
    try:
        save_chat(session.get("username"), clean_markdown(user_query), bot_text)
    except Exception:
        current_app.logger.exception("Failed to save chat in ajax_chat")

    return jsonify({
        "ok": True,
        "username": session.get("username"),
        "user_text": user_query,
        "bot_text": bot_text,
        "tts_audio_url": tts_url,   # 👈 now included
        "timestamp": datetime.utcnow().isoformat()
    })



def save_chat(username, user_message, bot_response):
    ch = ChatHistory(username=username, user_message=user_message, bot_response=bot_response)
    db.session.add(ch)
    db.session.commit()
    return ch.id

def get_recent_chats_for_user(username, limit=100):
    rows = ChatHistory.query.filter_by(username=username).order_by(ChatHistory.id.desc()).limit(limit).all()
    return [r.to_dict() for r in rows]



@app.route("/quick_questions", methods=["GET"])
def quick_questions():
    """
    Return a small list of quick question suggestions personalized for the logged-in user.
    Strategy:
      - If user has history, include their most recent queries (up to 3).
      - Add a few helpful defaults.
    Returns JSON: { ok: True, suggestions: ["...", "..."] }
    """
    if "username" not in session:
        defaults = [
            "What diseases affect wheat crops?",
            "How to improve soil quality?",
            "Which crops suit black soil?"
        ]
        return jsonify({"ok": True, "suggestions": defaults})

    username = session.get("username")
    try:
        history = get_recent_chats_for_user(username, limit=20)
    except Exception:
        history = []

    # Collect recent unique user queries (preserve order)
    seen = set()
    recent_queries = []
    for h in history:
        um = (h.get("user_message") or "").strip()
        if not um:
            continue
        # sanitize and limit length for pill text
        um_short = (um[:120] + '...') if len(um) > 120 else um
        if um_short not in seen:
            seen.add(um_short)
            recent_queries.append(um_short)
        if len(recent_queries) >= 3:
            break

    defaults = [
        "What diseases affect wheat crops?",
        "How to improve soil quality?",
        "Which crops suit black soil?"
    ]

    # If we have recent queries, personalize by prepending them
    suggestions = (recent_queries[:3] + defaults) if recent_queries else defaults

    # Ensure we return unique values and limit to e.g. 6 suggestions
    unique_sugg = []
    for s in suggestions:
        if s not in unique_sugg:
            unique_sugg.append(s)
        if len(unique_sugg) >= 6:
            break

    # Escape to be safe (client will render text)
    unique_sugg = [html.escape(s) for s in unique_sugg]

    return jsonify({"ok": True, "suggestions": unique_sugg})
# ----------------- Helper ML functions -----------------
def predict_soil(img_pil):
    """Predict soil type using Keras model from PIL Image (if model loaded)."""
    if soil_model is None:
        return "ModelUnavailable"
    img_resized = img_pil.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = soil_model.predict(img_array)
    index = int(np.argmax(pred))
    if index < len(soil_classes):
        return soil_classes[index]
    return f"class_{index}"

# ----------------- Routes -----------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/crops", methods=["GET", "POST"])
def crops():
    crop_suggestions = []
    if request.method == "POST":
        soil_type = request.form.get("soil_type", "")
        crop_suggestions = soil_crop_dict.get(soil_type.strip().lower(), ["Tomato", "Maize", "Chili"])
    return render_template("crops.html", crop_suggestions=crop_suggestions)

@app.route("/soil", methods=["GET", "POST"])
def soil():
    predicted_soil = None
    health_status = None
    ai_advice = None
    recommended_crops = []

    # defensive: ensure mapping exists, else fallback to empty dict
    mapping = globals().get("soil_crop_dict", {})
    if not isinstance(mapping, dict):
        # if someone accidentally overwrote the variable, recover to empty dict
        mapping = {}

    if request.method == "POST":
        file = request.files.get("soil_image")
        captured = request.form.get("captured_image")

        img = None
        if file:
            try:
                img = Image.open(file).convert("RGB")
            except Exception as e:
                flash(f"Error reading uploaded image: {e}", "danger")
        elif captured:
            try:
                img_data = captured.split(",")[1]
                img = Image.open(BytesIO(base64.b64decode(img_data))).convert("RGB")
            except Exception as e:
                flash(f"Error reading camera image: {e}", "danger")

        if img:
            predicted_soil = predict_soil(img)  # returns label from model or "ModelUnavailable"
            # normalize to lowercase string for lookup
            key = str(predicted_soil).strip().lower()
            # lookup in mapping, fallback to helpful default
            recommended_crops = mapping.get(key, ["No suggestions found"])
            # run yolo analysis / advice
            img_array = np.array(img)
            detected, health_status, ai_advice, annotation_url = analyze_image_array(img_array)


    return render_template(
        "soil.html",
        predicted_soil=predicted_soil,
        health_status=health_status,
        ai_advice=ai_advice,
        recommended_crops=recommended_crops,
        annotation_url=locals().get('annotation_url', None)
    )

@app.route("/plant", methods=["GET", "POST"])
def plant():
    """
    Upload / camera capture flow:
      - Run YOLO to detect leaf; crop highest-scoring box
      - If YOLO yields no boxes, run classifier on the full image
      - Return only: Status, Disease (or 'nil'), Recommendation, Confidence and annotation_url
    """
    status = None
    disease = None
    recommendation = None
    confidence = 0.0
    annotation_url = None
    detected_label = None

    if request.method == "POST":
        file = request.files.get("plant_image")
        captured = request.form.get("captured_plant_image")

        img = None
        filename = None

        # read image (uploaded or camera data URI)
        if file:
            try:
                img = Image.open(file).convert("RGB")
                # save for serving
                UPLOAD_DIR = Path("static/uploads")
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                filename = secure_filename(file.filename) or f"plant_{int(time.time()*1000)}.jpg"
                saved_path = UPLOAD_DIR / filename
                img.save(saved_path, quality=85)
            except Exception as e:
                flash(f"Error reading uploaded image: {e}", "danger")
                img = None

        elif captured:
            try:
                img_data = captured.split(",")[1]
                img = Image.open(BytesIO(base64.b64decode(img_data))).convert("RGB")
                # save
                UPLOAD_DIR = Path("static/uploads")
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                filename = f"plant_{int(time.time()*1000)}.jpg"
                saved_path = UPLOAD_DIR / filename
                img.save(saved_path, quality=85)
            except Exception as e:
                flash(f"Error reading camera image: {e}", "danger")
                img = None

        if img:
            img_array = np.array(img)

            # Run YOLO analysis (if YOLO loaded)
            try:
                top_label, yolo_status, ai_advice, annotation_url = analyze_image_array(img_array, conf_thresh=0.20, imgsz=640)
            except Exception as e:
                app.logger.exception("YOLO error in /plant")
                top_label, yolo_status, ai_advice, annotation_url = "Unknown", "General", "", None

            # If YOLO has boxes, analyze the top-most box by cropping (we rely on analyze_image_array to create annotation_url).
            # We will instead request the YOLO model directly to get coords so we can crop; fallback to full image classification.
            crop_img = None
            try:
                if yolo_model is not None:
                    res = yolo_model.predict(img_array, conf=0.20, imgsz=640)
                    if res and len(res) > 0:
                        boxes = getattr(res[0], "boxes", None)
                        if boxes is not None and len(boxes.xyxy) > 0:
                            # choose highest confidence box
                            confs = [float(c) for c in boxes.conf]
                            best = int(np.argmax(confs))
                            x1, y1, x2, y2 = map(int, map(float, boxes.xyxy[best]))
                            # ensure coords are inside image bounds
                            h, w = img_array.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            if x2 - x1 > 10 and y2 - y1 > 10:
                                crop_arr = img_array[y1:y2, x1:x2]
                                crop_img = Image.fromarray(crop_arr).convert("RGB")
                # else: no YOLO -> crop_img stays None
            except Exception:
                app.logger.exception("Error retrieving YOLO boxes for crop; falling back to full image")
                crop_img = None

            # Choose image for classification: use crop if available else full image
            img_for_classify = crop_img if crop_img is not None else img

            # Classify with leaf_model if available
            if leaf_model is None:
                # model missing — return YOLO-based guess if available, else friendly error
                status = yolo_status or "General"
                disease = top_label if top_label and top_label.lower() != "unknown" else "nil"
                recommendation = ai_advice or "No model available for specific pesticide recommendation."
                confidence = 0.0
            else:
                try:
                    # resize to model input and preprocess like training
                    _img = img_for_classify.resize(LEAF_IMG_SIZE).convert("RGB")
                    arr = np.array(_img) / 255.0
                    arr = np.expand_dims(arr, axis=0)
                    probs = leaf_model.predict(arr)[0]
                    top_idx = int(np.argmax(probs))
                    top_prob = float(probs[top_idx])
                    label = leaf_class_map.get(str(top_idx), f"class_{top_idx}")

                    # Normalize label/disease name
                    if label.lower().strip() in ("healthy", "diseased_healthy", "healthy_leaf"):
                        status = "Healthy"
                        disease = "nil"
                        recommendation = PESTICIDE_RECOMMENDATIONS.get("healthy", "No treatment required.")
                    else:
                        status = "Diseased"
                        # prettify name
                        disease = label.replace("diseased_", "").replace("_", " ").title()
                        # find recommended string by original label if available
                        recommendation = PESTICIDE_RECOMMENDATIONS.get(label, PESTICIDE_RECOMMENDATIONS.get("healthy"))
                    confidence = top_prob

                    # if low confidence — suppress disease name and mark nil
                    if status == "Diseased" and confidence < LEAF_CONF_THRESH:
                        disease = "nil"
                        recommendation = f"Low confidence ({confidence:.2f}). Please provide clearer picture or multiple images."
                except Exception as e:
                    app.logger.exception("Leaf classifier error")
                    status = yolo_status or "General"
                    disease = top_label if top_label and top_label.lower() != "unknown" else "nil"
                    recommendation = "Classification failed: " + str(e)

            # prepare annotation url to show in UI (we saved original file earlier)
            if filename:
                annotation_url = url_for('static', filename=f'uploads/{filename}', _external=False)

    return render_template(
        "plant.html",
        detected_label=disease if disease and disease != "nil" else None,
        health_status=status,
        ai_advice=recommendation,
        annotation_url=annotation_url,
        confidence=confidence
    )


@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if "username" not in session:
        flash("Please login to use the chatbot and view your chat history.", "warning")
        return redirect(url_for("login"))

    response_text = ""
    username = session.get("username")

    if request.method == "POST":
        user_query = request.form.get("user_query", "").strip()
        if user_query:
            try:
            # ✅ Use shared reply helper instead of direct Gemini call
                 lang_code = session.get("language") or "en"
                 response_text = generate_bot_reply(user_query, username, lang_code=lang_code)
            except Exception:
               current_app.logger.exception("Gemini generation error")
               response_text = "Gemini Error: could not generate a reply."

            # Save chat in DB
            try:
                safe_user_query = clean_markdown(user_query)
                safe_bot_text = response_text or ""
                save_chat(username, safe_user_query, safe_bot_text)
            except Exception:
                current_app.logger.exception("Error saving chat")

    # Load recent chat history
    history = []
    try:
        history = get_recent_chats_for_user(username, limit=100)
        for h in history:
            if h.get("user_message"):
                h["user_message"] = clean_markdown(h["user_message"])
            if h.get("bot_response"):
                h["bot_response"] = clean_markdown(h["bot_response"])
    except Exception:
        current_app.logger.exception("Error loading chat history")

    return render_template("chatbot.html", response_text=response_text, history=history)

@app.route('/asr', methods=['POST'])
def asr():
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify(ok=False, error="No audio file"), 400

    # Save temp
    temp_path = os.path.join("uploads", secure_filename(audio_file.filename))
    audio_file.save(temp_path)

    # Convert to WAV and run whisper
    try:
        wav_path = temp_path + ".wav"
        webm_to_wav(temp_path, wav_path, ffmpeg_exec=FFMPEG_EXEC)
        size = os.path.getsize(wav_path)
        model_name = "medium" if size < 1_000_000 else "base" 
        transcript, lang_code = transcribe_with_whisper(
            wav_path,
            model_name=model_name,
            language_hint=session.get("language")
        )
    except Exception as e:
        current_app.logger.exception("ASR error")
        transcript, lang_code = "Could not transcribe audio.", "en"

    # ✅ Generate reply using unified helper
    bot_response = generate_bot_reply(transcript, session.get("username"), lang_code=lang_code)

    # ✅ Add TTS here
    tts_url = None
    try:
        if GTTS_AVAILABLE and bot_response:
            from gtts import gTTS
            tts = gTTS(bot_response, lang=(lang_code or "en"))
            filename = f"tts_{int(time.time()*1000)}.mp3"
            path = os.path.join(TTS_DIR, filename)
            tts.save(path)
            tts_url = url_for("static", filename=f"tts/{filename}", _external=False)
    except Exception:
        current_app.logger.exception("TTS generation failed")

    # Save chat in DB
    try:
        ch = ChatHistory(
            username=session.get("username"),
            user_message=clean_markdown(transcript),
            bot_response=clean_markdown(bot_response),
            timestamp=datetime.utcnow()
        )
        db.session.add(ch)
        db.session.commit()
    except Exception:
        current_app.logger.exception("Failed to save chat in /asr")

    # ✅ Return JSON including TTS URL
    return jsonify({
        "ok": True,
        "user_text": transcript,
        "bot_text": bot_response,
        "tts_audio_url": tts_url,
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/contact")
def contact():
    return render_template("contact.html")

# -----------------SIGNUP + LOGIN + PROFILE -----------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()
        if not username or not password:
            flash("Username and password are required", "danger")
            return render_template("signup.html", states=list(STATES_DISTRICTS.keys()))

        if User.query.filter_by(username=username).first():
            flash("Username already exists", "danger")
            return render_template("signup.html", states=list(STATES_DISTRICTS.keys()))

        # Create user instance first, then set hashed password
        user = User(
            username=username,
            display_name=request.form.get("display_name") or username,
            email=request.form.get("email"),
            phone=request.form.get("phone"),
            state=request.form.get("state"),
            district=request.form.get("district"),
            language=request.form.get("language", "en"),
            land_size=request.form.get("land_size"),
            farming_type=request.form.get("farming_type"),
            profile_pic=None,
        )
        try:
            user.set_password(password)
        except Exception:
            user.password = generate_password_hash(password)

        db.session.add(user)
        db.session.commit()
        flash("Sign up successful — please log in.", "success")
        return redirect(url_for("login"))

    # GET: pass states to template so dropdown is populated
    return render_template("signup.html", states=list(STATES_DISTRICTS.keys()))

@app.route("/oauth/google")
def oauth_google():
    flash("Google signup not yet implemented. Please use manual signup for now.", "warning")
    return redirect(url_for("signup"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session["username"] = user.username
            flash("Logged in", "success")
            return redirect(url_for("profile"))

        flash("Invalid username or password", "danger")
    return render_template("login.html")


@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "username" not in session:
        return redirect(url_for("login"))

    user = User.query.filter_by(username=session["username"]).first()
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("login"))

    if request.method == "POST":
        # update details
        user.email = request.form.get("email") or user.email
        user.phone = request.form.get("phone") or user.phone
        user.state = request.form.get("state") or user.state
        user.district = request.form.get("district") or user.district
        user.language = request.form.get("language") or user.language
        user.land_size = request.form.get("land_size") or user.land_size
        user.farming_type = request.form.get("farming_type") or user.farming_type

        # handle profile picture upload
        file = request.files.get("profile_pic")
        if file and file.filename:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(PROFILE_PICS, filename)
            file.save(upload_path)
            user.profile_pic = f"profile_pics/{filename}"

        try:
            db.session.commit()
            flash("Profile updated successfully.", "success")
            return redirect(url_for("profile"))
        except Exception as e:
            db.session.rollback()
            flash(f"Unable to update profile: {e}", "danger")

    # ✅ Provide states list to the template
    states_list = list(STATES_DISTRICTS.keys())

    return render_template("profile.html", user_profile=user, states=states_list)

@app.route("/clear_history", methods=["POST"])
def clear_history():
    if "username" not in session:
        return {"ok": False}, 401
    ChatHistory.query.filter_by(username=session["username"]).delete()
    db.session.commit()
    return {"ok": True}

# required imports (put near top of app.py with other imports)
import requests
from werkzeug.utils import secure_filename

# where to store uploaded profile pics
PROFILE_PICS = os.path.join(BASE_DIR, "static", "profile_pics")
os.makedirs(PROFILE_PICS, exist_ok=True)

# small sample mapping of states -> districts (expand later or load from file)
# you should replace this with a more complete mapping JSON (states_and_districts.json)
STATES_DISTRICTS = {
   
  "Andhra Pradesh": ["Anantapur","Chittoor","East Godavari","Guntur","Krishna","Kurnool","Nellore","Prakasam","Srikakulam","Visakhapatnam","Vizianagaram","West Godavari","YSR Kadapa"],
  "Karnataka": ["Bagalkot","Ballari","Belagavi","Bangalore Rural","Bangalore Urban","Bidar","Chamarajanagar","Chikkaballapur","Chikmagalur","Chitradurga","Dakshina Kannada","Davanagere","Dharwad","Gadag","Hassan","Haveri","Kalaburagi","Kodagu","Kolar","Koppal","Mandya","Mysuru","Raichur","Ramanagara","Shivamogga","Tumakuru","Udupi","Uttara Kannada","Vijayapura","Yadgir"],
  "Kerala": ["Alappuzha","Ernakulam","Idukki","Kannur","Kasaragod","Kollam","Kottayam","Kozhikode","Malappuram","Palakkad","Pathanamthitta","Thiruvananthapuram","Thrissur","Wayanad"],
  "Tamil Nadu": ["Ariyalur","Chengalpattu","Chennai","Coimbatore","Cuddalore","Dharmapuri","Dindigul","Erode","Kallakurichi","Kanchipuram","Kanyakumari","Karur","Krishnagiri","Madurai","Mayiladuthurai","Nagapattinam","Namakkal","Perambalur","Pudukkottai","Ramanathapuram","Ranipet","Salem","Sivaganga","Tenkasi","Thanjavur","The Nilgiris","Theni","Thiruvallur","Thiruvarur","Thoothukudi","Tiruchirappalli","Tirunelveli","Tirupathur","Tiruppur","Tiruvannamalai","Vellore","Viluppuram","Virudhunagar"],
  "Telangana": ["Adilabad","Bhadradri Kothagudem","Hyderabad","Jagtial","Jangaon","Jayashankar Bhupalpally","Jogulamba Gadwal","Kamareddy","Karimnagar","Khammam","Komaram Bheem Asifabad","Mahabubabad","Mahabubnagar","Mancherial","Medak","Medchal–Malkajgiri","Mulugu","Nagarkurnool","Nalgonda","Nirmal","Nizamabad","Peddapalli","Rajanna Sircilla","Ranga Reddy","Sangareddy","Siddipet","Suryapet","Vikarabad","Wanaparthy","Warangal Rural","Warangal Urban","Yadadri Bhuvanagiri"],
  "Puducherry (UT)": ["Karaikal","Mahe","Puducherry","Yanam"]
   
}

@app.route("/get_districts", methods=["GET"])
def get_districts():
    state = request.args.get("state", "").strip()
    districts = STATES_DISTRICTS.get(state, [])
    return jsonify({"ok": True, "districts": districts})

@app.route("/upload_profile_pic", methods=["POST"])
def upload_profile_pic():
    if "username" not in session:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401

    user = User.query.filter_by(username=session["username"]).first()
    if not user:
        return jsonify({"ok": False, "error": "user_not_found"}), 404

    f = request.files.get("profile_pic")
    if not f:
        return jsonify({"ok": False, "error": "no_file"}), 400

    filename = secure_filename(f.filename)
    ts = int(time.time())
    ext = os.path.splitext(filename)[1] or ".jpg"
    out_name = f"{session['username']}_{ts}{ext}"
    out_path = os.path.join(PROFILE_PICS, out_name)
    f.save(out_path)

    user.profile_pic = f"profile_pics/{out_name}"
    db.session.commit()

    return jsonify({"ok": True, "url": url_for("static", filename=f"profile_pics/{out_name}")})

@app.route("/save_profile", methods=["POST"])
def save_profile():
    if "username" not in session:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401

    user = User.query.filter_by(username=session["username"]).first()
    if not user:
        return jsonify({"ok": False, "error": "user_not_found"}), 404

    data = request.form.to_dict()
    user.display_name = data.get("display_name") or user.display_name
    user.email = data.get("email") or user.email
    user.phone = data.get("phone") or user.phone
    user.language = data.get("language") or user.language
    user.state = data.get("state") or user.state
    user.district = data.get("district") or user.district
    user.land_size = data.get("land_size") or user.land_size
    user.farming_type = data.get("farming_type") or user.farming_type

    db.session.commit()
    return jsonify({"ok": True})


@app.route('/profile/edit', methods=['GET', 'POST'])
def edit_profile():
    if 'username' not in session:
        flash("Please log in to edit your profile.", "warning")
        return redirect(url_for('login'))

    user = User.query.filter_by(username=session['username']).first()
    if not user:
        flash("User not found.", "danger")
        return redirect(url_for('profile'))

    if request.method == 'POST':
        user.display_name = request.form.get('display_name') or user.display_name
        user.age = request.form.get('age') or user.age
        user.email = request.form.get('email') or user.email
        user.phone = request.form.get('phone') or user.phone
        user.district = request.form.get('district') or user.district
        user.state = request.form.get('state') or user.state
        user.language = request.form.get('language') or user.language
        user.land_size = request.form.get('land_size') or user.land_size
        user.farming_type = request.form.get('farming_type') or user.farming_type

        file = request.files.get('profile_pic')
        if file and file.filename:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(PROFILE_PICS, filename)
            file.save(upload_path)
            user.profile_pic = f"profile_pics/{filename}"

        try:
            db.session.commit()
            flash("Profile updated successfully.", "success")
            return redirect(url_for('profile'))
        except Exception:
            db.session.rollback()
            flash("Unable to update profile. Please try again.", "danger")

    return render_template('edit_profile.html', user_profile=user)

# Weather endpoint (OpenWeatherMap current weather by city)
OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")

@app.route("/get_weather", methods=["GET"])
def get_weather():
    # query param 'district' expected
    district = request.args.get("district", "").strip()
    if not district:
        return jsonify({"ok": False, "error": "no_district"}), 400
    if not OPENWEATHER_KEY:
        return jsonify({"ok": False, "error": "no_weather_key"}), 500

    # call OpenWeatherMap current weather (city name) - could be improved by geocoding
    params = {"q": district, "appid": OPENWEATHER_KEY, "units": "metric"}
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        # extract summary fields to return
        weather = {
            "ok": True,
            "name": data.get("name"),
            "temp_c": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_m_s": data["wind"]["speed"]
        }
        return jsonify(weather)
    except Exception as e:
        current_app.logger.exception("Weather API error")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


def predict_leaf_disease(pil_image):
    """
    pil_image: PIL.Image (RGB)
    returns: (pred_label, confidence)
    """
    global leaf_model, leaf_class_map
    if leaf_model is None:
        return None, 0.0
    img = pil_image.resize((224,224)).convert("RGB")
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = leaf_model.predict(arr)[0]
    idx = int(np.argmax(probs))
    label = leaf_class_map.get(str(idx), str(idx))
    return label, float(probs[idx])

# Load leaf disease model (optional)
# ----------------- Load leaf disease classifier -----------------
import tensorflow as tf

LEAF_MODEL_FILE = "leaf_disease_mobilenet_finetuned.h5"   # <- change if needed
CLASS_MAP_FILE = "class_map.json"
LEAF_IMG_SIZE = (128, 128)   # model input size
LEAF_CONF_THRESH = 0.30      # below this, treat disease as 'nil' (low confidence)

leaf_model = None
leaf_class_map = {}

# Pesticide / treatment recommendations — edit to fit local/regional guidance
PESTICIDE_RECOMMENDATIONS = {
    "diseased_blight": "Apply copper-based fungicide at label rate. Remove severely infected leaves. Repeat every 7–14 days.",
    "diseased_blackrot": "Use systemic fungicide and prune infected parts. Improve airflow.",
    "diseased_leaf_curl_virus": "Viral disease — remove infected plants. Control aphid/whitefly vectors (insecticidal soap, sticky traps).",
    "diseased_leaf_scorch": "Likely environmental — adjust watering and reduce stress. No pesticide usually required.",
    "diseased_leafspot": "Use protectant fungicide (copper or chlorothalonil). Sanitation and removing infected leaves helps.",
    "diseased_mosaic_virus": "Virus — remove infected plants, use virus-free seedlings, control vectors.",
    "diseased_powdery_mildew": "Apply sulfur or potassium bicarbonate; consider systemic fungicide for severe cases.",
    "diseased_rust": "Use contact or systemic fungicide and remove infected debris. Improve airflow.",
    "healthy": "No treatment needed. Continue good cultural practices."
}

try:
    if os.path.isfile(LEAF_MODEL_FILE):
        leaf_model = tf.keras.models.load_model(LEAF_MODEL_FILE)
        print(f"Loaded leaf model from {LEAF_MODEL_FILE}")
    else:
        print(f"Leaf model file not found: {LEAF_MODEL_FILE}")
except Exception as e:
    print("Could not load leaf model:", e)
    leaf_model = None

try:
    if os.path.isfile(CLASS_MAP_FILE):
        with open(CLASS_MAP_FILE, "r", encoding="utf-8") as fh:
            leaf_class_map = json.load(fh)
        print(f"Loaded class map from {CLASS_MAP_FILE}")
    else:
        print(f"Class map file not found: {CLASS_MAP_FILE}")
except Exception as e:
    print("Could not load class_map.json:", e)
    leaf_class_map = {}



UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/upload_user_file", methods=["POST"])
def upload_user_file():
    if "username" not in session:
        return jsonify({"ok": False, "error": "not_logged_in"}), 401
    f = request.files.get("file")
    if not f:
        return jsonify({"ok": False, "error": "no_file"}), 400
    user_dir = os.path.join(UPLOAD_DIR, session["username"])
    os.makedirs(user_dir, exist_ok=True)
    filename = secure_filename(f.filename)
    path = os.path.join(user_dir, filename)
    f.save(path)
    return jsonify({"ok": True, "path": path})  

# ----------------- Run -----------------
if __name__ == "__main__":
    app.run(debug=True)
