"""
Web UI and API for the RAG pipeline.
Features: streaming answers, conversation history, file upload (PDF/DOCX/TXT/JSON/MD/CSV),
evaluation endpoint, health check, and index stats.
"""

from flask import Flask, request, jsonify, render_template_string, Response
import json, os, webbrowser, threading, uuid, time
from werkzeug.utils import secure_filename
from rag import (
    load_docs, chunk_docs, flatten, VectorStore,
    parse_pdf, parse_docx, LLM_MODEL,
)
from groq import Groq

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------
_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env):
    for line in open(_env, encoding="utf-8-sig"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("Starting up...")
docs = load_docs()
chunks = chunk_docs(docs)
store = VectorStore(chunks)
print("Ready.\n")

# In-memory conversation sessions: session_id -> [{"role": ..., "content": ...}]
conversations = {}
MAX_HISTORY = 10  # max turns to keep per session

# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  :root {
    --bg: #0f1117; --surface: #1a1c25; --border: #2a2d3a;
    --text: #e4e4e7; --text-dim: #8b8d98;
    --accent: #7c6df0; --accent-dim: rgba(124,109,240,0.12);
    --green: #34d399; --green-dim: rgba(52,211,153,0.12);
    --user-bg: rgba(124,109,240,0.06);
  }
  body { font-family: 'Inter', system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; align-items: center; }
  main { width: 100%; max-width: 640px; padding: 0 20px; display: flex; flex-direction: column; min-height: 100vh; }

  .header { padding-top: min(18vh, 140px); text-align: center; transition: padding-top 0.4s ease; }
  .header.pushed { padding-top: 40px; }
  .header h1 { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.75rem; font-weight: 600; letter-spacing: -0.02em; margin-bottom: 6px; }
  .header p { font-size: 0.85rem; color: var(--text-dim); }

  .search-area { margin-top: 28px; position: relative; }
  .search-area input { width: 100%; padding: 14px 52px 14px 18px; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; color: var(--text); font-size: 0.95rem; font-family: inherit; outline: none; transition: border-color 0.2s; }
  .search-area input:focus { border-color: var(--accent); }
  .search-area input::placeholder { color: var(--text-dim); }
  .search-area button { position: absolute; right: 6px; top: 50%; transform: translateY(-50%); width: 38px; height: 38px; border-radius: 8px; border: none; background: var(--accent); color: #fff; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: opacity 0.2s; }
  .search-area button:disabled { opacity: 0.4; cursor: default; }
  .search-area button svg { width: 18px; height: 18px; }

  .toolbar { margin-top: 12px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .tool-btn { font-size: 0.78rem; color: var(--text-dim); background: var(--surface); border: 1px dashed var(--border); border-radius: 8px; padding: 7px 14px; cursor: pointer; transition: border-color 0.2s, color 0.2s; font-family: inherit; }
  .tool-btn:hover { border-color: var(--accent); color: var(--text); }
  .tool-btn.danger:hover { border-color: #f87171; color: #f87171; }
  .upload-status { font-size: 0.78rem; color: var(--green); opacity: 0; transition: opacity 0.3s; }
  .upload-status.show { opacity: 1; }

  .drop-overlay { display: none; position: fixed; inset: 0; background: rgba(124,109,240,0.08); border: 2px dashed var(--accent); z-index: 100; align-items: center; justify-content: center; }
  .drop-overlay.active { display: flex; }
  .drop-overlay span { font-size: 1.1rem; color: var(--accent); background: var(--surface); padding: 16px 32px; border-radius: 12px; }

  .chat-area { margin-top: 24px; flex: 1; padding-bottom: 60px; display: flex; flex-direction: column; gap: 12px; }

  .msg { border-radius: 12px; padding: 16px; animation: fadeUp 0.3s ease; }
  .msg.user { background: var(--user-bg); border: 1px solid var(--border); font-size: 0.88rem; color: var(--text-dim); }
  .msg.assistant { background: var(--surface); border: 1px solid var(--border); }
  .msg .answer-text { font-size: 0.92rem; line-height: 1.7; white-space: pre-wrap; }
  .msg .sources { margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--border); display: flex; flex-wrap: wrap; gap: 6px; }
  .msg .sources span { font-size: 0.72rem; color: var(--accent); background: var(--accent-dim); padding: 3px 10px; border-radius: 20px; font-weight: 500; }
  .cursor { display: inline-block; width: 2px; height: 1em; background: var(--accent); vertical-align: text-bottom; animation: blink 0.8s step-end infinite; }
  @keyframes blink { 50% { opacity: 0; } }
  @keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
</style>
</head>
<body>

<div class="drop-overlay" id="drop-overlay"><span>Drop files to add to knowledge base</span></div>

<main>
  <div class="header" id="header">
    <h1>Ask anything</h1>
    <p>Answers grounded in your documents</p>
  </div>

  <div class="search-area">
    <input type="text" id="q" placeholder="Type a question..." autocomplete="off" />
    <button id="btn" onclick="ask()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
    </button>
  </div>

  <div class="toolbar">
    <button class="tool-btn" onclick="document.getElementById('file-input').click()">+ Add documents</button>
    <input type="file" id="file-input" multiple accept=".txt,.md,.json,.csv,.pdf,.docx" hidden />
    <button class="tool-btn danger" onclick="clearChat()">Clear chat</button>
    <span class="upload-status" id="upload-status"></span>
  </div>

  <div class="chat-area" id="chat-area"></div>
</main>

<script>
const q = document.getElementById('q');
const btn = document.getElementById('btn');
const chatArea = document.getElementById('chat-area');
const header = document.getElementById('header');
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');
const dropOverlay = document.getElementById('drop-overlay');

let sessionId = crypto.randomUUID ? crypto.randomUUID() : Date.now().toString();

q.addEventListener('keydown', e => { if (e.key === 'Enter' && q.value.trim()) ask(); });
fileInput.addEventListener('change', () => uploadFiles(fileInput.files));

let dragCount = 0;
document.addEventListener('dragenter', e => { e.preventDefault(); dragCount++; dropOverlay.classList.add('active'); });
document.addEventListener('dragleave', e => { e.preventDefault(); dragCount--; if (dragCount <= 0) { dragCount = 0; dropOverlay.classList.remove('active'); } });
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => {
  e.preventDefault(); dragCount = 0; dropOverlay.classList.remove('active');
  if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
});

async function uploadFiles(files) {
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  uploadStatus.textContent = 'Uploading...';
  uploadStatus.classList.add('show');
  try {
    const res = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    uploadStatus.textContent = data.message;
    setTimeout(() => uploadStatus.classList.remove('show'), 3000);
  } catch {
    uploadStatus.textContent = 'Upload failed';
    setTimeout(() => uploadStatus.classList.remove('show'), 3000);
  }
  fileInput.value = '';
}

function clearChat() {
  chatArea.innerHTML = '';
  sessionId = crypto.randomUUID ? crypto.randomUUID() : Date.now().toString();
  header.classList.remove('pushed');
  fetch('/clear', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({session_id: sessionId}) });
}

async function ask() {
  const query = q.value.trim();
  if (!query) return;
  btn.disabled = true;
  header.classList.add('pushed');

  // Show user message
  chatArea.insertAdjacentHTML('beforeend', `<div class="msg user">${escHtml(query)}</div>`);

  // Show assistant placeholder
  const assistantDiv = document.createElement('div');
  assistantDiv.className = 'msg assistant';
  assistantDiv.innerHTML = '<div class="answer-text"><span class="cursor"></span></div>';
  chatArea.appendChild(assistantDiv);
  assistantDiv.scrollIntoView({behavior: 'smooth'});

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, session_id: sessionId})
    });

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let text = '';
    let sources = [];
    const textEl = assistantDiv.querySelector('.answer-text');

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      const chunk = dec.decode(value);
      for (const line of chunk.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);
        if (payload === '[DONE]') continue;
        try {
          const msg = JSON.parse(payload);
          if (msg.token) { text += msg.token; textEl.innerHTML = escHtml(text) + '<span class="cursor"></span>'; }
          if (msg.sources) sources = msg.sources;
        } catch {}
      }
    }
    text = text.replace(/\(Source[s]?\s*\d[\d,\s&and]*\)/gi, '').replace(/\[Source[s]?\s*\d[\d,\s&and]*\]/gi, '').trim();
    textEl.innerHTML = escHtml(text);
    if (sources.length) {
      const chips = [...new Set(sources)].map(s => `<span>${escHtml(s)}</span>`).join('');
      assistantDiv.insertAdjacentHTML('beforeend', `<div class="sources">${chips}</div>`);
    }
  } catch (e) {
    assistantDiv.querySelector('.answer-text').textContent = 'Something went wrong. Check that GROQ_API_KEY is set.';
  }
  btn.disabled = false;
  q.value = '';
  assistantDiv.scrollIntoView({behavior: 'smooth'});
}

function escHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/health")
def health():
    """Health check for deployment platforms."""
    return jsonify({"status": "ok", "timestamp": time.time()})


@app.route("/stats")
def stats():
    """Index statistics."""
    return jsonify(store.stats())


@app.route("/clear", methods=["POST"])
def clear():
    """Clear a conversation session."""
    sid = request.json.get("session_id", "")
    if sid in conversations:
        del conversations[sid]
    return jsonify({"cleared": True})


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"message": "No files received"}), 400

    new_docs = []
    errors = []

    for f in files:
        filename = secure_filename(f.filename)
        raw = f.read()
        text = ""

        try:
            if filename.lower().endswith(".pdf"):
                text = parse_pdf(raw)
            elif filename.lower().endswith(".docx"):
                text = parse_docx(raw)
            elif filename.lower().endswith(".json"):
                data = json.loads(raw.decode("utf-8", errors="ignore"))
                if isinstance(data, list):
                    for doc in data:
                        content = doc.get("content", "")
                        if isinstance(content, dict):
                            content = flatten(content)
                        new_docs.append({
                            "id": doc.get("id", str(uuid.uuid4())[:8]),
                            "title": doc.get("title", filename),
                            "content": content,
                        })
                    continue
                else:
                    text = json.dumps(data, indent=2)
            else:
                text = raw.decode("utf-8", errors="ignore")
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
            continue

        if text.strip():
            new_docs.append({
                "id": str(uuid.uuid4())[:8],
                "title": filename.rsplit(".", 1)[0],
                "content": text,
            })

    if not new_docs:
        msg = "No readable content found"
        if errors:
            msg += f" ({'; '.join(errors)})"
        return jsonify({"message": msg}), 400

    new_chunks = chunk_docs(new_docs)
    store.add_chunks(new_chunks)

    n = len(new_docs)
    msg = f"Added {n} document{'s' if n != 1 else ''} ({len(new_chunks)} chunks)"
    if errors:
        msg += f". Errors: {'; '.join(errors)}"
    return jsonify({"message": msg})


@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("query", "")
    session_id = request.json.get("session_id", "default")

    if not query:
        return jsonify({"answer": "Please enter a question.", "sources": []})

    # Retrieve relevant chunks
    results = store.search(query)
    sources = [r["title"] for r in results if r["score"] > 0.3]

    context_parts = []
    for i, chunk in enumerate(results, 1):
        context_parts.append(f"[Source {i}: {chunk['title']}]\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    # Build conversation messages
    history = conversations.get(session_id, [])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer based only on the provided context. "
                "Be concise. If the context doesn't contain the answer, say so."
            ),
        },
    ]

    # Add conversation history (limited)
    for turn in history[-(MAX_HISTORY * 2):]:
        messages.append(turn)

    # Add current turn with context
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}",
    })

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def stream():
        yield f"data: {json.dumps({'sources': sources})}\n\n"

        full_answer = ""
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=1024,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                full_answer += delta.content
                yield f"data: {json.dumps({'token': delta.content})}\n\n"

        # Save to conversation history
        if session_id not in conversations:
            conversations[session_id] = []
        conversations[session_id].append({"role": "user", "content": query})
        conversations[session_id].append({"role": "assistant", "content": full_answer})

        # Trim history
        if len(conversations[session_id]) > MAX_HISTORY * 2:
            conversations[session_id] = conversations[session_id][-(MAX_HISTORY * 2):]

        yield "data: [DONE]\n\n"

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Evaluate retrieval quality on test questions.

    POST JSON body:
    {
      "questions": [
        {"query": "What is Firefly's best light cone?", "expected_sources": ["Firefly"]},
        ...
      ],
      "top_k": 5
    }

    Or POST with no body to use the built-in eval set (data/eval.json if it exists).
    """
    data = request.get_json(silent=True) or {}
    top_k = data.get("top_k", 5)

    questions = data.get("questions")
    if not questions:
        eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "eval.json")
        if os.path.exists(eval_path):
            with open(eval_path, "r", encoding="utf-8") as f:
                questions = json.load(f)
        else:
            return jsonify({"error": "No questions provided and no data/eval.json found."}), 400

    results = []
    hits = 0

    for q in questions:
        query = q["query"]
        expected = [s.lower() for s in q.get("expected_sources", [])]

        retrieved = store.search(query, n_results=top_k)
        retrieved_titles = [r["title"].lower() for r in retrieved]

        # Check if any expected source appears in retrieved titles (partial match)
        found = any(
            any(exp in title for title in retrieved_titles)
            for exp in expected
        )
        if found:
            hits += 1

        results.append({
            "query": query,
            "expected": q.get("expected_sources", []),
            "retrieved": [{"title": r["title"], "score": round(r["score"], 3)} for r in retrieved[:3]],
            "hit": found,
        })

    total = len(questions)
    accuracy = hits / total if total > 0 else 0

    return jsonify({
        "accuracy": round(accuracy, 3),
        "hits": hits,
        "total": total,
        "results": results,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    if os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RENDER"):
        # Deployed: don't open browser, bind 0.0.0.0
        app.run(host="0.0.0.0", port=port)
    else:
        threading.Timer(1, webbrowser.open, args=[f"http://localhost:{port}"]).start()
        app.run(port=port)
