"""Build the search index from the PDF.  Usage: python build_index.py path/to/book.pdf"""
import re, sys, json, subprocess, pickle
from sklearn.feature_extraction.text import TfidfVectorizer

pdf = sys.argv[1] if len(sys.argv) > 1 else "KML_Medical_book.pdf"
raw = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout
pages = raw.split("\f")

SECTIONS = ["Definition", "Description", "Purpose", "Precautions", "Preparation", "Aftercare",
            "Risks", "Causes and symptoms", "Diagnosis", "Treatment", "Alternative treatment",
            "Allopathic treatment", "Expected results", "Prognosis", "Prevention", "Normal results",
            "Abnormal results", "Resources"]
sec_re = re.compile(r"^(%s)\s*$" % "|".join(SECTIONS))


BYLINE = re.compile(r"^[A-Z][a-z]+( [A-Z]\.)* [A-Z][A-Za-z'-]+(,? (MD|M\.D\.|PhD|Ph\.D\.|RN|R\.N\.|M\.S\.|DrPH|JD|J\.D\.))*$")
def good_title(t):
    if not t or len(t) > 70 or len(t) < 3: return False
    if not (t[0].isupper() or t[0].isdigit()): return False
    if t[-1] in ".,;:" or " see " in t or "•" in t or "Inc" in t: return False
    if t in SECTIONS or t.upper() == t and len(t) > 25: return False
    if BYLINE.match(t): return False
    if len(t.split()) > 6 or "—" in t or ". " in t or re.search(r"PharmD|PhD|MD$|Institute|Press", t): return False
    if t.isupper() and len(t) > 4: return False
    if len(t.split()) == 3 and all(w[0].isupper() for w in t.split()): return False
    return True

# Join all pages, remembering page number for each line
lines = []
for pno, ptxt in enumerate(pages, 1):
    for ln in ptxt.split("\n"):
        ln = ln.rstrip()
        if re.fullmatch(r"\d{1,4}", ln.strip()) or ln.startswith("GALE ENCYCLOPEDIA OF MEDICINE"):
            continue
        lines.append((pno, ln))

# An entry starts at a title line immediately followed (after blank) by "Definition"
entries, cur = [], None
for i, (pno, ln) in enumerate(lines):
    if ln.strip() == "Definition":
        j = i - 1
        while j >= 0 and not lines[j][1].strip():
            j -= 1
        title = ""
        for back in range(6):               # look back a few non-empty lines for a plausible title
            while j >= 0 and not lines[j][1].strip():
                j -= 1
            if j < 0: break
            cand = lines[j][1].strip()
            if good_title(cand):
                title = cand; break
            j -= 1
        if title:
            cur = {"title": title, "page": pno, "start": j + 1, "sections": {}}
            entries.append(cur)
            cur["_def_idx"] = i

chunks = []
for k, e in enumerate(entries):
    end = entries[k + 1]["start"] - 1 if k + 1 < len(entries) else len(lines)
    body = lines[e["_def_idx"]:end]
    sec, buf = "Overview", []
    def flush():
        txt = " ".join(x for x in buf if x.strip())
        txt = re.sub(r"(\w)- (\w)", r"\1\2", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt and sec != "Resources" and len(txt) > 40:
            chunks.append({"title": e["title"], "section": sec, "page": body[0][0], "text": txt})
    for pno, ln in body:
        m = sec_re.match(ln.strip())
        if m:
            flush(); sec, buf = m.group(1), []
        else:
            buf.append(ln)
    flush()

# Split very long sections into ~900-char passages so answers stay readable
final = []
for c in chunks:
    t = c["text"]
    if len(t) <= 1100:
        final.append(c); continue
    sents = re.split(r"(?<=[.!?])\s+", t)
    piece = ""
    for s in sents:
        if len(piece) + len(s) > 900 and piece:
            final.append({**c, "text": piece.strip()}); piece = ""
        piece += s + " "
    if piece.strip():
        final.append({**c, "text": piece.strip()})

docs = [f"{c['title']} {c['title']} {c['section']} {c['text']}" for c in final]
vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=400000)
X = vec.fit_transform(docs)
titles = sorted({c["title"] for c in final})
pickle.dump({"chunks": final, "vec": vec, "X": X, "titles": titles}, open("data/index.pkl", "wb"))
print(f"{len(entries)} entries, {len(final)} passages indexed")
