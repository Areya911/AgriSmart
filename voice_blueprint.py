# voice_blueprint.py
import os
import time
import tempfile
from flask import Blueprint, request, jsonify, current_app, url_for
from werkzeug.utils import secure_filename

# helpers from speech_utils
from speech_utils import resolve_ffmpeg, webm_to_wav, transcribe_with_whisper, create_tts_gtts, normalize_lang_code

voice_bp = Blueprint("voice_bp", __name__)

# allow the app to override this before registering blueprint
FFMPEG_EXEC = resolve_ffmpeg()

# directory (under project root) to save tts files
TTS_DIR_NAME = os.path.join("static", "tts")


@voice_bp.route("/asr", methods=["POST"])
def asr_endpoint():
    """
    Receives form 'audio' file and optional 'preferred_lang'.
    Returns JSON: { user_text, text, lang, tts_audio_url }
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    f = request.files['audio']
    preferred_lang = (request.form.get('preferred_lang') or 'auto').strip()
    ts = int(time.time() * 1000)
    tmpdir = tempfile.gettempdir()
    in_name = f"upload_{ts}_{secure_filename(f.filename or 'speech.webm')}"
    in_path = os.path.join(tmpdir, in_name)
    wav_path = os.path.join(tmpdir, f"upload_{ts}.wav")

    try:
        # save uploaded file
        f.save(in_path)
    except Exception as e:
        current_app.logger.exception("Failed to save uploaded audio")
        return jsonify({"error": "Failed to save upload", "detail": str(e)}), 500

    try:
        # ensure ffmpeg present (use blueprint-level FFMPEG_EXEC or resolve again)
        ff = FFMPEG_EXEC or resolve_ffmpeg()
        if not ff:
            raise RuntimeError("ffmpeg not found on server. Install ffmpeg or set FFMPEG_EXEC.")

        # convert to wav
        webm_to_wav(ff, in_path, wav_path)


        # transcribe with whisper (may raise)
        try:
            detected_text, detected_lang = transcribe_with_whisper(wav_path, model_name="small")
        except Exception as te:
            current_app.logger.exception("Whisper transcribe failed")
            return jsonify({"error": "Transcription failed", "detail": str(te)}), 500

        # normalize language code (preferred_lang like 'ta-IN', detected_lang like 'ta')
                # normalize language code (preferred_lang like 'ta-IN', detected_lang like 'ta')
        final_lang = normalize_lang_code(preferred_lang, detected_lang)

        current_app.logger.debug("ASR: detected_text=%s", detected_text[:400])
        current_app.logger.debug("ASR: detected_lang=%s, preferred_lang=%s => final_lang=%s", detected_lang, preferred_lang, final_lang)

        # call the app-level LLM helper if available
        bot_reply = ""
        try:
            if hasattr(current_app, "query_gemini_reply"):
                current_app.logger.debug("voice_blueprint calling current_app.query_gemini_reply")
                bot_reply = current_app.query_gemini_reply(detected_text, final_lang)
                current_app.logger.debug("voice_blueprint got bot_reply (first 400 chars): %s", (bot_reply or "")[:400])
            else:
                current_app.logger.warning("current_app.query_gemini_reply not found")
        except Exception as e:
            current_app.logger.exception("Error calling current_app.query_gemini_reply: %s", e)
            bot_reply = ""


        # generate TTS if available and reply text is present
        tts_url = None
        try:
            if bot_reply and 'create_tts_gtts' in globals():
                # ensure tts directory exists
                tts_dir = os.path.join(os.getcwd(), TTS_DIR_NAME)
                os.makedirs(tts_dir, exist_ok=True)
                # create tts file (create_tts_gtts maps lang short code e.g. 'hi','ta')
                tts_path = create_tts_gtts(bot_reply, final_lang[:2], tts_dir)
                tts_fn = os.path.basename(tts_path)
                tts_url = url_for('static', filename=f"tts/{tts_fn}", _external=False)
        except Exception:
            current_app.logger.exception("TTS generation failed")
            tts_url = None

        return jsonify({
            "user_text": detected_text,
            "text": bot_reply,
            "lang": final_lang,
            "tts_audio_url": tts_url
        })

    except RuntimeError as re:
        current_app.logger.exception("ASR pipeline runtime error")
        return jsonify({"error": "Processing failed", "detail": str(re)}), 500
    except Exception as e:
        current_app.logger.exception("ASR pipeline error")
        return jsonify({"error": "Unexpected server error", "detail": str(e)}), 500
    finally:
        # cleanup temporary files
        for p in (in_path, wav_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
