# speech_utils.py
import os
import subprocess
import shutil
import time

# Optional runtime imports to avoid failing if library missing
try:
    import whisper
    WHISPER_AVAILABLE = True
    _whisper_model = None
except Exception:
    WHISPER_AVAILABLE = False
    _whisper_model = None

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# Resolve an ffmpeg executable (returns path or None)
def resolve_ffmpeg(fallback_paths=None):
    ff = shutil.which("ffmpeg")
    if ff and os.path.isfile(ff):
        return ff
    # optional fallback candidates
    if fallback_paths is None:
        fallback_paths = [
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
        ]
    for p in fallback_paths:
        if os.path.isfile(p):
            return p
    return None

# Convert webm (or other) into 16k mono WAV using ffmpeg
def webm_to_wav(ffmpeg_exec, in_path, out_path, sample_rate=16000):
    if not ffmpeg_exec:
        raise RuntimeError("ffmpeg not found (ffmpeg_exec is None). Please install ffmpeg or set correct path.")
    cmd = [
        ffmpeg_exec,
        "-y",
        "-i", in_path,
        "-ar", str(sample_rate),
        "-ac", "1",
        out_path
    ]
    # run and raise if failed
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors='ignore') if proc.stderr else ""
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {stderr}")
    return out_path

# Lazy-load whisper model
_whisper_model = None

def ensure_whisper_model(model_name="small"):
    global _whisper_model
    if not WHISPER_AVAILABLE:
        return None
    if _whisper_model is None:
        # load lazily once
        _whisper_model = whisper.load_model(model_name)
    return _whisper_model

# Transcribe wav with whisper; returns (text, lang)
def transcribe_with_whisper(wav_path, model_name="small"):
    model = ensure_whisper_model(model_name)
    if model is None:
        raise RuntimeError("Whisper model not available. Install 'whisper'.")
    result = model.transcribe(wav_path, language=None, task="transcribe")
    text = result.get("text", "").strip()
    lang = result.get("language") or result.get("lang") or None
    if not lang:
        lang = "en"
    return text, lang


# Simple helper to create TTS (gTTS) file; returns path
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

