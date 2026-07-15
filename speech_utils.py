# speech_utils.py
import os
import subprocess
import shutil
import time

try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

_whisper_model = None

def resolve_ffmpeg(fallback_paths=None):
    ff = shutil.which("ffmpeg")
    if ff and os.path.isfile(ff):
        return ff
    if fallback_paths is None:
        fallback_paths = [
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Users\USER\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe",
        ]
    for p in fallback_paths:
        if os.path.isfile(p):
            return p
    return None

def webm_to_wav(ffmpeg_exec, in_path, out_path):
    ff = ffmpeg_exec or resolve_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg executable not found.")
    # Normalize, remove silence, denoise, and downsample 16k mono
    cmd = [
        ff, '-y', '-i', in_path,
        '-af', 'silenceremove=1:0:-50dB,highpass=f=80,lowpass=f=8000,afftdn,loudnorm',
        '-ar', '16000', '-ac', '1',
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path

def ensure_whisper_model(model_name="small"):
    global _whisper_model
    if not WHISPER_AVAILABLE:
        return None
    if _whisper_model is None:
        _whisper_model = whisper.load_model(model_name)
    return _whisper_model

def transcribe_with_whisper(wav_path, model_name="small", language_hint=None):
    model = ensure_whisper_model(model_name)
    if model is None:
        raise RuntimeError("Whisper model not available. Install 'whisper'.")

    # Farming keywords to help Whisper bias recognition
    farming_prompt = (
        "agriculture, farming, soil, crop, irrigation, fertilizer, pesticide, "
        "wheat, paddy, rice, maize, millet, groundnut, cotton, sugarcane, "
        "disease, blight, rust, sow, alluvial, black, red, clay, leaf spot, pest control, organic, yield, harvest"
    )

    options = {
        "language": language_hint or None,
        "task": "transcribe",
        "initial_prompt": farming_prompt
    }

    result = model.transcribe(wav_path, **options)
    text = result.get("text", "").strip()
    lang = result.get("language") or result.get("lang") or "en"
    return text, lang

def create_tts_gtts(text, lang, out_dir, filename=None):
    if not GTTS_AVAILABLE:
        raise RuntimeError("gTTS not installed. `pip install gTTS` to enable TTS.")
    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        filename = f"tts_{int(time.time() * 1000)}.mp3"
    out_path = os.path.join(out_dir, filename)
    short = (lang or "en")[:2]
    tts = gTTS(text=text, lang=short)
    tts.save(out_path)
    return out_path

PREFERRED_TO_SHORT = {
    'auto': None,
    'en-US': 'en', 'en-IN': 'en',
    'hi-IN': 'hi', 'hi': 'hi',
    'ta-IN': 'ta', 'ta': 'ta',
    'te-IN': 'te', 'te': 'te',
    'ml-IN': 'ml', 'ml': 'ml',
    'kn-IN': 'kn', 'kn': 'kn'
}

def normalize_lang_code(preferred_lang, detected_lang):
    short = None
    if preferred_lang:
        short = PREFERRED_TO_SHORT.get(preferred_lang, None)
    if not short and detected_lang:
        short = (detected_lang or 'en')[:2]
    return short or 'en'
