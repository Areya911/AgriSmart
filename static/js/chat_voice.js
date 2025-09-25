// static/js/chat_voice.js
// AgriSmart voice flow client script
// Expects elements in chatbot.html:
//  #start-voice, #stop-voice, #preferred-lang, #transcript-text, #bot-audio, .chat-history

(() => {
  const startBtn = document.getElementById('start-voice');
  const stopBtn = document.getElementById('stop-voice');
  const preferredLang = document.getElementById('preferred-lang');
  const transcriptText = document.getElementById('transcript-text');
  const botAudio = document.getElementById('bot-audio');
  const chatHistoryContainer = document.querySelector('.chat-history');

  let mediaRecorder = null;
  let recordedChunks = [];
  let activeStream = null;

  // Helpers
  function $log(...args) {
    console.debug('[chat_voice]', ...args);
  }

  function escapeHtml(s) {
    if (!s) return '';
    return s.replace(/[&<>"']/g, m =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[m]
    );
  }

  function cleanMarkdown(s) {
    if (!s) return '';
    try {
      let t = String(s);
      t = t.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
      t = t.replace(/```[\s\S]*?```/g, '');
      t = t.replace(/^[ \t]{0,3}#{1,6}\s+/gm, '');
      t = t.replace(/^[ \t]*[\*\-\+]\s+/gm, '');
      t = t.replace(/^[ \t]*\d+\.\s+/gm, '');
      t = t.replace(/\*\*(.*?)\*\*/gs, '$1');
      t = t.replace(/__(.*?)__/gs, '$1');
      t = t.replace(/\*(.*?)\*/gs, '$1');
      t = t.replace(/_(.*?)_/gs, '$1');
      t = t.replace(/`(.*?)`/gs, '$1');
      t = t.replace(/\*{1,3}/g, '');
      t = t.split('\n').map(l => l.replace(/\s+$/, '')).join('\n').replace(/\n{3,}/g, '\n\n');
      return t.trim();
    } catch (e) {
      console.warn('cleanMarkdown failed', e);
      return s;
    }
  }

  function prependHistoryItem(item) {
    try {
      if (!chatHistoryContainer) return;
      const el = document.createElement('div');
      el.className = 'chat-entry';
      el.style.marginBottom = '12px';
      el.style.paddingBottom = '8px';
      el.style.borderBottom = '1px dashed #eee';
      el.innerHTML = `
        <div style="font-size:0.9em; color:#333;">
          <strong>${escapeHtml(item.username || 'You')}</strong>
          <span style="color:#666; font-size:0.8em; margin-left:8px;">
            ${escapeHtml(item.timestamp || new Date().toISOString())}
          </span>
        </div>
        <div style="margin-top:6px;">
          <div style="font-weight:600; color:#0b6e3a;">You:</div>
          <div style="white-space:pre-wrap; margin:4px 0 6px 6px;">
            ${escapeHtml(cleanMarkdown(item.user_message || item.user_text || ''))}
          </div>
          <div style="font-weight:600; color:#2a4a8f;">Bot:</div>
          <div style="white-space:pre-wrap; margin:4px 0 0 6px; color:#222;">
            ${escapeHtml(cleanMarkdown(item.bot_response || item.bot_text || item.text || ''))}
          </div>
        </div>
      `;
      chatHistoryContainer.insertBefore(el, chatHistoryContainer.firstChild);
    } catch (e) {
      console.error('prependHistoryItem', e);
    }
  }

  async function ensureMicrophone() {
    try {
      return await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      alert('Microphone access denied or unavailable. Please allow microphone access.');
      throw err;
    }
  }

  function startRecording(stream) {
    recordedChunks = [];
    activeStream = stream;
    let opts = {};
    try {
      mediaRecorder = new MediaRecorder(stream, opts);
    } catch (e) {
      mediaRecorder = new MediaRecorder(stream);
    }
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.start();
    startBtn.disabled = true;
    stopBtn.disabled = false;
    transcriptText.innerText = "(Listening...)";
    $log('Recording started');
  }

  function stopRecorderAndGetBlob() {
    return new Promise((resolve, reject) => {
      if (!mediaRecorder) return reject(new Error('No active recorder'));

      const fallbackTimeout = setTimeout(() => {
        try {
          const blob = new Blob(recordedChunks, { type: recordedChunks[0]?.type || 'audio/webm' });
          cleanup();
          transcriptText.innerText = "(Stopped — fallback)";
          resolve(blob);
        } catch (err) {
          reject(err);
        }
      }, 3000);

      mediaRecorder.onstop = () => {
        clearTimeout(fallbackTimeout);
        try {
          const blob = new Blob(recordedChunks, { type: recordedChunks[0]?.type || 'audio/webm' });
          cleanup();
          transcriptText.innerText = "(Stopped)";
          resolve(blob);
        } catch (err) {
          reject(err);
        }
      };

      try {
        mediaRecorder.stop();
      } catch (e) {
        clearTimeout(fallbackTimeout);
        const blob = new Blob(recordedChunks, { type: recordedChunks[0]?.type || 'audio/webm' });
        cleanup();
        transcriptText.innerText = "(Stopped)";
        resolve(blob);
      }
    });
  }

  function cleanup() {
    mediaRecorder = null;
    recordedChunks = [];
    if (activeStream) {
      try { activeStream.getTracks().forEach(t => t.stop()); } catch (e) {}
      activeStream = null;
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }

  async function postToAsr(blob) {
    const form = new FormData();
    form.append('audio', blob, 'speech.webm');
    form.append('preferred_lang', preferredLang.value || 'auto');

    try {
      const resp = await fetch('/asr', { method: 'POST', body: form });
      if (!resp.ok) {
        let bodyText = '';
        try { bodyText = await resp.json(); } catch (e) { bodyText = await resp.text(); }
        throw new Error('Server returned ' + resp.status + ': ' + (typeof bodyText === 'string' ? bodyText : JSON.stringify(bodyText)));
      }
      return resp.json();
    } catch (err) {
      transcriptText.innerText = `(Error: ${err.message || err})`;
      console.error('postToAsr error', err);
      throw err;
    }
  }

  async function saveChatToServer(user_text, bot_text) {
    try {
      const resp = await fetch('/save_chat_ajax', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_text: user_text || '', bot_text: bot_text || '' })
      });
      if (!resp.ok) {
        const txt = await resp.text();
        $log('save_chat_ajax failed', txt);
        return null;
      }
      return resp.json();
    } catch (e) {
      $log('saveChatToServer error', e);
      return null;
    }
  }

  function extractUserAndBot(result) {
    const userCandidates = ['user_text', 'transcript', 'user_transcript', 'input_text', 'user_message'];
    const botCandidates = ['text', 'bot_text', 'bot_response', 'reply'];

    let user_text = null;
    for (const k of userCandidates) {
      if (result && result[k]) { user_text = result[k]; break; }
    }

    let bot_text = null;
    for (const k of botCandidates) {
      if (result && result[k]) { bot_text = result[k]; break; }
    }

    if (!user_text && result && typeof result === 'object') {
      if (result.data && result.data.user_text) user_text = result.data.user_text;
    }

    return { user_text: user_text || '', bot_text: bot_text || '' };
  }

  // === UI events ===
  startBtn.addEventListener('click', async () => {
    try {
      const stream = await ensureMicrophone();
      startRecording(stream);
    } catch (e) {
      console.error('start error', e);
    }
  });

  stopBtn.addEventListener('click', async () => {
  try {
    // get recorded blob
    const blob = await stopRecorderAndGetBlob();

    // UI: disable buttons while we process
    startBtn.disabled = true;
    stopBtn.disabled = true;
    transcriptText.innerText = "(Uploading audio...)";

    // Step 1: Transcribe only (POST to /asr)
    const form = new FormData();
    form.append('audio', blob, 'speech.webm');
    form.append('preferred_lang', preferredLang.value || 'auto');

    const asrResp = await fetch('/asr', { method: 'POST', body: form });
    if (!asrResp.ok) {
      const body = await (asrResp.text().catch(()=>'')); 
      throw new Error(`ASR failed: ${asrResp.status} ${body}`);
    }
    const asrResult = await asrResp.json();

    // try to extract a transcript from possible keys
    const transcriptRaw = asrResult.user_text || asrResult.transcript || asrResult.text || '';
    const transcript = cleanMarkdown(transcriptRaw || '').trim();
    transcriptText.innerText = transcript || '(No transcription returned)';

    // If no transcript, stop early and save nothing to chat endpoint
    if (!transcript) {
      // Re-enable buttons and bail out
      startBtn.disabled = false;
      stopBtn.disabled = true;
      return;
    }

    // Step 2: Push transcript into /ajax_chat to get bot reply + TTS + save
    transcriptText.innerText = "(Getting reply...)";

    const chatResp = await fetch('/ajax_chat', {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'X-Requested-With': 'XMLHttpRequest'
      },
      body: JSON.stringify({ user_query: transcript })
    });

    if (!chatResp.ok) {
      const txt = await (chatResp.text().catch(()=>'')); 
      throw new Error(`Chat endpoint failed: ${chatResp.status} ${txt}`);
    }

    const data = await chatResp.json();

    // Step 3: handle reply + tts url
    const botText = cleanMarkdown(data.bot_text || data.bot_response || data.text || '').trim();

    // Set transcript display and show reply
    transcriptText.innerText = transcript;

    // show play controls if TTS URL returned
    const ttsUrl = data.tts_audio_url || null;
    if (ttsUrl) {
      try {
        botAudio.src = ttsUrl;
        botAudio.style.display = 'block';
        // attempt play (may be blocked by browser)
        await botAudio.play().catch(e => {
          console.warn('Audio autoplay blocked', e);
        });
      } catch (e) {
        console.warn('TTS playback failed', e);
      }
    }

    // add to UI history (prepend)
    prependHistoryItem({
      username: window.sessionUser || 'You',
      user_message: transcript,
      bot_response: botText,
      timestamp: new Date().toISOString()
    });

  } catch (err) {
    console.error('Voice flow error', err);
    transcriptText.innerText = '(Error: ' + (err.message || err) + ')';

    // show an error entry in history so user sees something
    try {
      prependHistoryItem({
        username: window.sessionUser || 'You',
        user_message: '(voice input)',
        bot_response: 'Error processing voice input. See console for details.',
        timestamp: new Date().toISOString()
      });
    } catch (e) { /* ignore */ }
  } finally {
    // restore UI state
    cleanup(); // re-enable start button and ensure streams stopped
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
});

  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && document.activeElement.tagName !== 'TEXTAREA') {
      e.preventDefault();
      if (!startBtn.disabled) startBtn.click();
      else if (!stopBtn.disabled) stopBtn.click();
    }
  });

})();
