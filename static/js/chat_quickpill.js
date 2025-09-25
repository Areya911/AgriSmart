// static/js/chat_quickpill.js
// Dynamic quick pills: load suggestions from /quick_questions and post to /ajax_chat
// New behavior: PREPEND new chats at the top (reverse-chronological).
// Also: optimistic placeholders are replaced by final server responses (no duplicates).

(function () {
  // Wait for DOM ready
  document.addEventListener('DOMContentLoaded', () => {
    const listContainer = document.querySelector('.qq-list');
    const historyContainer = document.querySelector('.chat-history');

    if (!listContainer) {
      console.warn('[quickpill] .qq-list not found in DOM');
      return;
    }
    if (!historyContainer) {
      console.warn('[quickpill] .chat-history not found in DOM — quickpill will still render but cannot show messages.');
    }

    console.log('[quickpill] init');

    // Fetch and render suggestions
    async function loadSuggestions() {
      try {
        const resp = await fetch('/quick_questions');
        if (!resp.ok) {
          console.warn('[quickpill] /quick_questions returned', resp.status);
          return;
        }
        const json = await resp.json();
        if (!json.ok || !Array.isArray(json.suggestions)) {
          console.warn('[quickpill] bad suggestions payload', json);
          return;
        }
        renderPills(json.suggestions);
      } catch (e) {
        console.error('[quickpill] loadSuggestions error', e);
      }
    }

    // Render pills inside .qq-list (replace existing children)
    function renderPills(suggestions) {
      listContainer.innerHTML = '';
      suggestions.forEach(s => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'qq-pill';
        btn.textContent = s;
        btn.style.cursor = 'pointer';
        listContainer.appendChild(btn);
      });
      console.log('[quickpill] rendered', suggestions.length, 'pills');
    }

    // Utility: escape html
    function escapeHtml(s) {
      if (!s) return '';
      return String(s).replace(/[&<>"']/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[m]);
    }

    // Create a unique optimistic id
    function makeOptimisticId() {
      return 'opt-' + Date.now().toString(36) + '-' + Math.floor(Math.random() * 9999).toString(36);
    }

    // Prefer a global prepend helper if available (your chat_voice.js may provide it)
    // If not present, use fallback that prepends DOM elements to .chat-history
    function getPrependFn() {
      if (typeof window.prependHistoryItem === 'function') return window.prependHistoryItem;

      // fallback prepend function: supports optimisticId if provided
      return function fallbackPrepend(saved, opts = {}) {
        if (!historyContainer) return saved;
        // If an optimisticId is provided and replace flag set, try to replace existing optimistic entry
        if (opts.optimisticId) {
          const exist = historyContainer.querySelector(`[data-optimistic-id="${opts.optimisticId}"]`);
          if (exist) {
            // replace content inside that optimistic node
            exist.innerHTML = buildChatInnerHTML(saved, opts.showMeta);
            // ensure newest visible at top
            historyContainer.scrollTop = 0;
            return saved;
          }
        }

        // build wrapper and prepend
        const wrapper = document.createElement('div');
        wrapper.className = 'chat-entry';
        if (opts.optimisticId) wrapper.setAttribute('data-optimistic-id', opts.optimisticId);
        wrapper.innerHTML = buildChatInnerHTML(saved, opts.showMeta);
        historyContainer.prepend(wrapper);
        historyContainer.scrollTop = 0;
        return saved;
      };
    }

    // small helper that creates the inner markup used by fallback
    function buildChatInnerHTML(saved, showMeta = true) {
      const metaHtml = showMeta ? `<div style="font-size:0.9em;color:#333;"><strong>${escapeHtml(saved.username || '')}</strong> <span style="color:#666;font-size:0.8em;margin-left:8px;">${escapeHtml(saved.timestamp || '')}</span></div>` : '';
      return `${metaHtml}
        <div style="margin-top:6px;">
          <div style="font-weight:600;color:#0b6e3a;">You:</div>
          <div style="white-space:pre-wrap;margin:4px 0 6px 6px;">${escapeHtml(saved.user_message || '')}</div>
          <div style="font-weight:600;color:#2a4a8f;">Bot:</div>
          <div style="white-space:pre-wrap;margin:4px 0 0 6px;color:#222;">${escapeHtml(saved.bot_response || '')}</div>
        </div>`;
    }

    const prependFn = getPrependFn();

    // Click handler (delegation) - handles both rendered pills and future ones
    document.body.addEventListener('click', async (ev) => {
      const t = ev.target;
      if (!t || !t.classList) return;
      if (!t.classList.contains('qq-pill')) return;

      ev.preventDefault();
      const qText = (t.textContent || '').trim();
      if (!qText) return;

      console.log('[quickpill] clicked:', qText);

      // Create optimistic placeholder and keep its id
      const optimisticId = makeOptimisticId();
      let optimisticInserted = false;
      try {
        if (prependFn) {
          prependFn({
            username: window.sessionUser || 'You',
            user_message: qText,
            bot_response: "(thinking...)",
            timestamp: new Date().toISOString()
          }, { optimisticId, showMeta: true });
          optimisticInserted = true;
        }
      } catch (e) {
        console.warn('[quickpill] optimistic prepend failed', e);
      }

      // call server (POST /ajax_chat)
      try {
        const resp = await fetch('/ajax_chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ user_query: qText })
        });

        if (!resp.ok) {
          const txt = await resp.text().catch(()=> '');
          console.error('[quickpill] /ajax_chat error', resp.status, txt);
          // Replace optimistic with an error or prepend new error
          if (optimisticInserted) {
            prependFn({
              username: window.sessionUser || 'You',
              user_message: qText,
              bot_response: `Error: server returned ${resp.status}`,
              timestamp: new Date().toISOString()
            }, { optimisticId, showMeta: true });
          } else {
            prependFn({
              username: window.sessionUser || 'You',
              user_message: qText,
              bot_response: `Error: server returned ${resp.status}`,
              timestamp: new Date().toISOString()
            });
          }
          return;
        }

        const json = await resp.json();
        if (!json.ok) {
          console.warn('[quickpill] server returned ok:false', json);
          prependFn({
            username: window.sessionUser || 'You',
            user_message: qText,
            bot_response: `Error: ${json.error || 'unknown'}`,
            timestamp: new Date().toISOString()
          }, { optimisticId, showMeta: true });
          return;
        }

        // Build saved object
        const saved = {
          username: json.username || window.sessionUser || 'You',
          user_message: json.user_text || qText,
          bot_response: json.bot_text || '',
          timestamp: json.timestamp || new Date().toISOString()
        };

        // Replace optimistic placeholder if present, otherwise prepend the final saved
        if (optimisticInserted) {
          prependFn(saved, { optimisticId, showMeta: true });
        } else {
          prependFn(saved, { showMeta: true });
        }

        // play tts if returned
        if (json.tts_audio_url) {
          try {
            // If there is an audio element with id 'bot-audio' use it, else create and play
            let audioEl = document.getElementById('bot-audio');
            if (!audioEl) {
              audioEl = document.createElement('audio');
              audioEl.id = 'bot-audio';
              audioEl.style.display = 'none';
              document.body.appendChild(audioEl);
            }
            audioEl.src = json.tts_audio_url;
            audioEl.play().catch(e => console.warn('TTS play blocked', e));
          } catch (e) { console.warn('play tts', e); }
        }

        // reload contextual suggestions
        await loadSuggestions();

      } catch (err) {
        console.error('[quickpill] network error', err);
        if (optimisticInserted) {
          prependFn({
            username: window.sessionUser || 'You',
            user_message: qText,
            bot_response: 'Network error — see console.',
            timestamp: new Date().toISOString()
          }, { optimisticId, showMeta: true });
        } else {
          prependFn({
            username: window.sessionUser || 'You',
            user_message: qText,
            bot_response: 'Network error — see console.',
            timestamp: new Date().toISOString()
          });
        }
      }
    });

    // initially load suggestions
    loadSuggestions();
  }); // DOMContentLoaded
})();
