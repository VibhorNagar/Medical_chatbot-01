const chat = document.getElementById('chat');
const form = document.getElementById('chat-form');
const input = document.getElementById('msg');
const sendBtn = document.getElementById('send');

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;       // textContent: no HTML injection
  return e;
}

function addBubble(who, cls = '') {
  const row = el('div', 'msg ' + who);
  const bubble = el('div', 'bubble ' + cls);
  row.appendChild(bubble);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
  return { row, bubble };
}

function passage(r) {
  const wrap = document.createElement('div');
  wrap.appendChild(el('div', 'meta', `PDF page ${r.page} · cosine score ${r.score.toFixed(3)}`));
  wrap.appendChild(el('div', '', r.text));
  return wrap;
}

function welcome() {
  const { bubble } = addBubble('bot');
  bubble.appendChild(el('div', '', 'Hi! Ask me about a medical topic from the book (it covers entries A to B).'));
  const chips = el('div', 'chips');
  ['What is asthma?', 'Symptoms of bronchitis', 'How is acne treated?', 'What is anthrax?'].forEach(q => {
    const b = el('button', 'chip', q);
    b.type = 'button';
    b.addEventListener('click', () => ask(q));
    chips.appendChild(b);
  });
  bubble.appendChild(chips);
}

async function ask(text) {
  text = text.trim();
  if (!text) return;
  input.value = '';
  addBubble('user').bubble.textContent = text;
  const typing = addBubble('bot', 'typing');
  typing.bubble.textContent = 'Searching the book…';
  sendBtn.disabled = true;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    typing.row.remove();
    const { bubble } = addBubble('bot', res.ok ? '' : 'error');
    if (!res.ok) {
      bubble.textContent = data.error || 'Something went wrong.';
    } else {
      if (data.emergency) bubble.appendChild(el('div', 'alert', '⚠️ If this is a medical emergency, call your local emergency number now.'));
      if (!data.matched) {
        bubble.appendChild(el('div', '', "I couldn't find a good match in the book. Try naming the condition (for example: asthma, acne, bronchitis). The book only covers entries A to B."));
      } else {
        bubble.appendChild(passage(data.results[0]));
        if (data.results.length > 1) {
          const d = document.createElement('details');
          d.appendChild(el('summary', '', `More passages (${data.results.length - 1})`));
          data.results.slice(1).forEach(r => { const p = passage(r); p.style.marginTop = '10px'; d.appendChild(p); });
          bubble.appendChild(d);
        }
      }
      bubble.appendChild(el('div', 'meta', data.disclaimer)).style.marginTop = '10px';
    }
  } catch (err) {
    typing.row.remove();
    addBubble('bot', 'error').bubble.textContent = 'Could not reach the server. Is it running?';
  } finally {
    sendBtn.disabled = false;
    input.focus();
    chat.scrollTop = chat.scrollHeight;
  }
}

form.addEventListener('submit', e => { e.preventDefault(); ask(input.value); });
welcome();
