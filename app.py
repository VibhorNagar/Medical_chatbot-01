"""Medical chatbot backend (Flask).  Answers come only from the indexed book."""
import os, re, pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from sklearn.metrics.pairwise import linear_kernel

BASE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(os.path.join(BASE, "data", "index.pkl"), "rb"))
CHUNKS, VEC, X, TITLES = D["chunks"], D["vec"], D["X"], D["titles"]
BY_TITLE = {}
for i, c in enumerate(CHUNKS):
    BY_TITLE.setdefault(c["title"], []).append(i)

app = Flask(__name__, static_folder="static")

INTENTS = [
    (r"symptom|sign|feel|cause|why do|why does|why am", ["Causes and symptoms"]),
    (r"treat|cure|medic|drug|therapy|remed|manage|help", ["Treatment", "Allopathic treatment", "Alternative treatment"]),
    (r"prevent|avoid|reduce risk|protect", ["Prevention"]),
    (r"diagnos|test|detect|screen|examin", ["Diagnosis"]),
    (r"prognosis|outlook|survive|recover|life expectancy|how long", ["Prognosis", "Expected results"]),
    (r"what is|what are|define|meaning|explain|tell me about|about", ["Definition", "Description"]),
]
EMERGENCY = re.compile(r"chest pain|can'?t breathe|cannot breathe|difficulty breathing|stroke|suicid|kill myself|overdose|unconscious|severe bleeding|poison", re.I)
DISCLAIMER = "This is general information from a medical reference book, not medical advice. See a doctor for diagnosis or treatment."

def norm(s): return re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("’", "'"))

def find_topic(q):
    """Return an entry title whose name appears in the question (longest match wins)."""
    nq = " " + norm(q) + " "
    best = None
    for t in TITLES:
        nt = norm(t).strip()
        if len(nt) >= 3 and f" {nt} " in nq and (best is None or len(nt) > len(norm(best))):
            best = t
    return best

def intent_sections(q):
    for pat, secs in INTENTS:
        if re.search(pat, q.lower()):
            return secs
    return []

def search(q, topic=None, k=40):
    qv = VEC.transform([q])
    sims = linear_kernel(qv, X).ravel()
    if topic:
        idx = BY_TITLE[topic]
        sims = sims.copy()
        sims[idx] += 0.5
    top = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in top if sims[i] > 0]

@app.post("/api/chat")
def chat():
    data = request.get_json(force=True)
    q = (data.get("message") or "").strip()
    last_topic = data.get("last_topic")
    if not q:
        return jsonify(error="Empty message"), 400

    topic = find_topic(q)
    followup = False
    if not topic and last_topic and len(q.split()) <= 8:
        topic, followup = last_topic, True          # "what about treatment?" style follow-ups
    wanted = intent_sections(q)
    hits = search(q + (" " + topic if followup else ""), topic)
    if not hits and not topic:
        return jsonify(answer="I couldn't find anything about that in the book. Try a disease, symptom, test or treatment name (for example “asthma”, “bronchitis symptoms”, “what is bulimia nervosa”).",
                       sources=[], related=[], topic=None, disclaimer=DISCLAIMER, emergency=bool(EMERGENCY.search(q)))

    # Pick the winning entry
    if topic:
        best_title = topic
    else:
        score = {}
        for i, s in hits[:15]:
            t = CHUNKS[i]["title"]; score[t] = score.get(t, 0) + s
        best_title = max(score, key=score.get)
    if not topic and hits and max(s for _, s in hits) < 0.08:
        return jsonify(answer="I'm not sure I have a good match for that in the book. Could you rephrase or name the condition?",
                       sources=[], related=[], topic=None, disclaimer=DISCLAIMER, emergency=bool(EMERGENCY.search(q)))

    idxs = BY_TITLE[best_title]
    sc = dict(hits)
    ranked = sorted(idxs, key=lambda i: (CHUNKS[i]["section"] in wanted) * 1.0 + sc.get(i, 0), reverse=True)
    chosen = ranked[:3] if wanted else ranked[:2]
    if not wanted:   # default to a definition-first overview
        defs = [i for i in idxs if CHUNKS[i]["section"] in ("Definition", "Description")][:2]
        chosen = defs or chosen
    chosen = sorted(chosen, key=lambda i: i)         # keep book order
    parts = [{"section": CHUNKS[i]["section"], "text": CHUNKS[i]["text"], "page": CHUNKS[i]["page"]} for i in chosen]

    related, seen = [], {best_title}
    for i, s in hits:
        t = CHUNKS[i]["title"]
        if t not in seen:
            seen.add(t); related.append(t)
        if len(related) == 4: break
    return jsonify(topic=best_title, parts=parts, related=related, followup=followup,
                   sections=sorted({CHUNKS[i]["section"] for i in idxs}),
                   disclaimer=DISCLAIMER, emergency=bool(EMERGENCY.search(q)))

@app.get("/api/topics")
def topics():
    return jsonify(TITLES)

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
