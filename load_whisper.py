# load_whisper.py
import whisper

print("Loading Whisper model 'tiny' (this will download weights if needed)...")
model = whisper.load_model("tiny")
print("Model loaded successfully. Detected device:", model.device)
print("Whisper is ready.")
