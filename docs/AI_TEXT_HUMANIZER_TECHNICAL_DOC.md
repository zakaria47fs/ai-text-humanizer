# AI Text Humanizer — Technical Documentation for Claude Code

## Project Overview

Build a web application that takes AI-generated text as input and outputs humanized text that passes Originality.ai detection as "Human" in at least 80% of cases.

**Key Constraints:**
- Must be your own solution (RAG-based system + prompt engineering pipeline) — NOT a wrapper around another humanizer API
- Output must be high quality, natural, readable, suitable for students
- Text must not be more than 30–40% longer than the original
- No major grammar issues
- Budget: up to $100 in API credits
- Stack: Python backend, simple web frontend
- Delivery: a working URL

---

## Architecture Overview

```
User pastes AI text
        │
        ▼
┌──────────────────┐
│   FastAPI Backend │
│                  │
│  1. Text Analysis │ ← Identify AI markers (uniform sentences, cliché phrases, low burstiness)
│        │         │
│        ▼         │
│  2. RAG Retrieval │ ← Pull similar human writing samples from vector DB (ChromaDB)
│        │         │
│        ▼         │
│  3. LLM Rewrite  │ ← Multi-step prompt pipeline using GPT-4o / Gemini
│        │         │
│        ▼         │
│  4. Post-Process  │ ← Length check, final cleanup
│        │         │
│        ▼         │
│  Return humanized │
│      text         │
└──────────────────┘
        │
        ▼
  Simple web UI (HTML/JS)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend Framework | FastAPI |
| LLM API | OpenAI (GPT-4o) and/or Google Gemini 2.5 Flash |
| RAG Vector Store | ChromaDB (lightweight, no external DB needed) |
| Embeddings | OpenAI `text-embedding-3-small` or `sentence-transformers` (local) |
| NLP Utilities | spaCy, NLTK (optional, for text analysis step) |
| Frontend | Single HTML page with vanilla JS (or minimal Jinja2 template) |
| Deployment | Railway, Render, or any platform that supports Python |

---

## Project Structure

```
ai-humanizer/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py             # API keys, settings
│   ├── routers/
│   │   └── humanize.py       # /humanize endpoint
│   ├── services/
│   │   ├── analyzer.py       # Step 1: Text analysis (detect AI patterns)
│   │   ├── rag.py            # Step 2: RAG retrieval from human writing corpus
│   │   ├── rewriter.py       # Step 3: LLM rewriting pipeline
│   │   └── postprocess.py    # Step 4: Length check, final cleanup
│   ├── prompts/
│   │   ├── system_prompt.py  # Main humanizer system prompt
│   │   ├── analysis_prompt.py # Prompt for AI pattern detection
│   │   └── refine_prompt.py  # Prompt for second-pass refinement
│   └── corpus/
│       ├── build_corpus.py   # Script to build the vector DB from human text samples
│       └── samples/          # Folder of human-written text files (.txt)
├── static/
│   └── index.html            # Frontend UI
├── requirements.txt
├── Dockerfile                # For deployment
└── README.md
```

---

## Step-by-Step Implementation Guide

### Step 1: Text Analyzer (`app/services/analyzer.py`)

Purpose: Scan the input text and identify specific AI writing patterns so the rewriter knows what to fix.

**What to detect:**
- Average sentence length and variance (low variance = AI fingerprint)
- Overused AI words (see banned word list below)
- Repetitive transition patterns ("Furthermore," "Moreover," "In conclusion")
- Uniform paragraph structure
- Excessive use of em dashes (—)
- "Not just X, but also Y" constructions
- Setup/conclusion phrases

**Implementation approach:**
```python
import re
from collections import Counter

def analyze_text(text: str) -> dict:
    """Analyze text for AI writing patterns. Returns a report dict."""
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentence_lengths = [len(s.split()) for s in sentences]
    
    # Burstiness: variance in sentence lengths
    avg_len = sum(sentence_lengths) / len(sentence_lengths)
    variance = sum((l - avg_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
    burstiness = variance ** 0.5  # standard deviation
    
    # Check for banned AI words
    ai_words = [
        "delve", "tapestry", "embark", "landscape", "realm", "furthermore",
        "moreover", "crucial", "pivotal", "nuanced", "multifaceted",
        "ever-evolving", "game-changer", "paradigm", "facilitate",
        "comprehensive", "intricate", "notably", "It's worth noting",
        "In today's", "In conclusion", "revolutionize", "groundbreaking",
        "harness", "foster", "leverage", "robust", "streamline",
        "cutting-edge", "holistic", "synergy", "illuminate", "underscore",
        "navigate", "elevate", "myriad", "testament", "encompasses",
        "spearhead", "burgeoning", "commendable", "meticulous",
        "proliferation", "propel", "remnant", "resonate", "poised"
    ]
    
    text_lower = text.lower()
    found_ai_words = [w for w in ai_words if w.lower() in text_lower]
    
    # Em dash count
    em_dash_count = text.count("—") + text.count("--")
    
    # Transition word repetition
    transitions = re.findall(
        r'\b(Furthermore|Moreover|Additionally|In addition|Consequently|'
        r'Nevertheless|However|Therefore|Thus|Hence|Indeed|Notably)\b',
        text, re.IGNORECASE
    )
    
    return {
        "sentence_count": len(sentences),
        "avg_sentence_length": round(avg_len, 1),
        "burstiness": round(burstiness, 2),
        "burstiness_is_low": burstiness < 5.0,  # AI typically has low burstiness
        "found_ai_words": found_ai_words,
        "ai_word_count": len(found_ai_words),
        "em_dash_count": em_dash_count,
        "repeated_transitions": dict(Counter(transitions)),
        "original_word_count": len(text.split()),
    }
```

---

### Step 2: RAG Retrieval (`app/services/rag.py`)

Purpose: Retrieve stylistically relevant human writing samples from a vector database to use as few-shot style references in the rewriting prompt.

**Why RAG matters here:**
- Grounds the LLM in actual human writing patterns instead of relying only on prompt instructions
- Provides concrete style examples the LLM can mimic
- Makes the system your "own solution" — this is the RAG part the client asked for

**Building the corpus:**
Collect 50–100 human-written text samples. Good sources:
- Student essays (public domain essay collections)
- Blog posts you write yourself or from public domain blogs
- Wikipedia "Good Article" excerpts (these read naturally)
- News articles (AP/Reuters style, which is clean and natural)
- Book excerpts from Project Gutenberg (modern, post-2000 if possible)

Each sample should be 200–500 words. Store them as .txt files in `app/corpus/samples/`.

**Implementation:**
```python
import chromadb
from chromadb.utils import embedding_functions
import os
import glob

# --- Build the corpus (run once) ---

def build_corpus(samples_dir: str, db_path: str = "./chroma_db"):
    """Build ChromaDB vector store from human text samples."""
    
    # Option A: Use OpenAI embeddings (better quality, costs a few cents)
    # ef = embedding_functions.OpenAIEmbeddingFunction(
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     model_name="text-embedding-3-small"
    # )
    
    # Option B: Use free local embeddings (no API cost)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="human_writing",
        embedding_function=ef
    )
    
    # Load all text files
    files = glob.glob(os.path.join(samples_dir, "*.txt"))
    documents = []
    ids = []
    metadatas = []
    
    for i, filepath in enumerate(files):
        with open(filepath, "r") as f:
            text = f.read().strip()
        documents.append(text)
        ids.append(f"sample_{i}")
        metadatas.append({"source": os.path.basename(filepath)})
    
    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    print(f"Added {len(documents)} samples to corpus.")
    return collection


# --- Query the corpus at runtime ---

def get_style_references(input_text: str, n_results: int = 3, db_path: str = "./chroma_db") -> list[str]:
    """Retrieve the most stylistically similar human writing samples."""
    
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(
        name="human_writing",
        embedding_function=ef
    )
    
    results = collection.query(
        query_texts=[input_text],
        n_results=n_results
    )
    
    return results["documents"][0]  # List of matching human text samples
```

---

### Step 3: LLM Rewriter (`app/services/rewriter.py`)

This is the core of the system. Uses a multi-step prompt pipeline.

**Pipeline:**
1. **Main Rewrite** — Structural + lexical humanization using the system prompt + RAG style references
2. **Refinement Pass** — Second LLM call to polish and catch remaining AI patterns
3. (Optional) **Cross-model rewrite** — Use a different LLM for the refinement pass (e.g., GPT-4o for rewrite, Gemini for refinement)

**Implementation:**
```python
import openai
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Alternative: Use Gemini ---
# import google.generativeai as genai
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# model = genai.GenerativeModel("gemini-2.5-flash")


def rewrite_text(
    original_text: str,
    analysis: dict,
    style_references: list[str]
) -> str:
    """Main rewriting pipeline."""
    
    # Format style references for the prompt
    style_block = "\n\n---\n\n".join(
        [f"STYLE REFERENCE {i+1}:\n{ref}" for i, ref in enumerate(style_references)]
    )
    
    # Format analysis findings
    issues = []
    if analysis["ai_word_count"] > 0:
        issues.append(f"AI vocabulary detected: {', '.join(analysis['found_ai_words'])}")
    if analysis["burstiness_is_low"]:
        issues.append(f"Low sentence variety (burstiness: {analysis['burstiness']})")
    if analysis["em_dash_count"] > 2:
        issues.append(f"Too many em dashes ({analysis['em_dash_count']})")
    if analysis["repeated_transitions"]:
        issues.append(f"Repetitive transitions: {analysis['repeated_transitions']}")
    
    issues_block = "\n".join(f"- {issue}" for issue in issues) if issues else "No major issues detected."
    
    # --- PASS 1: Main rewrite ---
    pass1_response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.85,  # Slightly high for more natural variation
        top_p=0.9,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Rewrite the following AI-generated text to sound naturally human-written.

DETECTED AI PATTERNS TO FIX:
{issues_block}

HUMAN WRITING STYLE REFERENCES (mimic this natural style):
{style_block}

ORIGINAL TEXT TO HUMANIZE:
{original_text}

CRITICAL RULES:
- Output must NOT be more than 35% longer than the original
- Original word count: {analysis['original_word_count']} words
- Maximum output: {int(analysis['original_word_count'] * 1.35)} words
- Preserve the original meaning completely
- Write at a natural reading level suitable for students
- Do NOT add any meta-commentary about the rewriting process"""}
        ]
    )
    
    pass1_text = pass1_response.choices[0].message.content
    
    # --- PASS 2: Refinement ---
    pass2_response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[
            {"role": "system", "content": REFINE_PROMPT},
            {"role": "user", "content": f"""Review and lightly refine this text. Fix any remaining AI-like patterns while keeping the natural human voice. Do NOT make it longer.

Maximum word count: {int(analysis['original_word_count'] * 1.35)} words

TEXT TO REFINE:
{pass1_text}"""}
        ]
    )
    
    return pass2_response.choices[0].message.content
```

---

### System Prompts (`app/prompts/`)

#### Main System Prompt (`system_prompt.py`)

```python
SYSTEM_PROMPT = """You are a text rewriter that transforms AI-generated content into natural human writing. Your output must be indistinguishable from text written by a real person.

WRITING RULES:

1. SENTENCE STRUCTURE — Vary sentence lengths dramatically. Mix short punchy sentences (5-8 words) with medium ones (12-18 words) and occasional long ones (25+ words). Real humans write unevenly. Never use more than two sentences of similar length in a row.

2. BANNED VOCABULARY — Never use these words or phrases under any circumstance:
   delve, tapestry, embark, landscape, realm, furthermore, moreover, crucial, pivotal, nuanced, multifaceted, ever-evolving, game-changer, paradigm, facilitate, comprehensive, intricate, notably, revolutionize, groundbreaking, harness, foster, leverage, robust, streamline, cutting-edge, holistic, synergy, illuminate, underscore, navigate, elevate, myriad, testament, encompasses, spearhead, burgeoning, commendable, meticulous, proliferation, propel, remnant, resonate, poised, It's worth noting, In today's digital age, In today's world, In conclusion, plays a crucial role, It is important to note, serves as a testament, a testament to, In the realm of, at the forefront

3. BANNED PUNCTUATION — Do not use em dashes (—). Use commas, periods, parentheses, or semicolons instead. Em dashes are a major AI fingerprint.

4. BANNED CONSTRUCTIONS:
   - "Not just X, but also Y" → rephrase naturally
   - "Whether it's X or Y" → rephrase
   - Starting paragraphs with "In a world where..." or "In an era of..."
   - Ending with generic calls to action
   - Lists of three with parallel structure (the AI triplet pattern)

5. NATURAL FLOW:
   - Start some sentences with "And", "But", "So", or "Because"
   - Use contractions naturally (don't, isn't, won't, it's)
   - Occasionally use informal transitions: "The thing is,", "Here's the deal:", "That said,"
   - Include occasional rhetorical questions
   - Use fragments for emphasis. Like this.
   - Vary paragraph lengths: some 1-2 sentences, some 4-5 sentences

6. VOICE:
   - Write in active voice primarily
   - Be direct and specific rather than abstract
   - Use concrete examples over general statements
   - Aim for a Flesch Reading Ease score around 60-70 (readable for students)
   - Sound like an informed person explaining something to a friend, not like a textbook

7. LENGTH:
   - Stay within 35% of the original text length
   - Do NOT pad with unnecessary words or filler
   - It's okay to be slightly shorter than the original

8. OUTPUT:
   - Return ONLY the rewritten text
   - No headers, labels, or meta-commentary
   - Do not mention that you rewrote anything"""
```

#### Refinement Prompt (`refine_prompt.py`)

```python
REFINE_PROMPT = """You are a final-pass editor. Your job is to review text and make small, targeted fixes to remove any remaining traces of AI writing patterns.

CHECK FOR:
- Any words from the AI vocabulary list (delve, tapestry, landscape, realm, etc.)
- Em dashes — replace with commas or periods
- Overly parallel sentence structures
- Sentences that all start the same way
- Paragraphs that are all the same length
- Text that sounds too "polished" or "smooth" — real writing has slight roughness

RULES:
- Make minimal changes. Don't rewrite everything.
- Keep the natural voice intact
- Don't make the text longer
- Return ONLY the refined text, no commentary"""
```

---

### Step 4: Post-Processor (`app/services/postprocess.py`)

```python
def postprocess(original_text: str, humanized_text: str, max_ratio: float = 1.35) -> dict:
    """Validate output meets length requirements and clean up."""
    
    original_words = len(original_text.split())
    humanized_words = len(humanized_text.split())
    ratio = humanized_words / original_words if original_words > 0 else 1.0
    
    # Clean up any extra whitespace
    humanized_text = " ".join(humanized_text.split())
    
    # Remove any accidental meta-commentary the LLM might have added
    lines = humanized_text.split("\n")
    cleaned_lines = [
        line for line in lines
        if not line.strip().lower().startswith(("here is", "here's the", "note:", "rewritten", "humanized"))
    ]
    humanized_text = "\n".join(cleaned_lines).strip()
    
    return {
        "text": humanized_text,
        "original_word_count": original_words,
        "humanized_word_count": humanized_words,
        "length_ratio": round(ratio, 2),
        "within_limit": ratio <= max_ratio,
        "warning": f"Text is {round((ratio - 1) * 100)}% longer than original" if ratio > max_ratio else None
    }
```

---

### FastAPI App (`app/main.py`)

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.services.analyzer import analyze_text
from app.services.rag import get_style_references
from app.services.rewriter import rewrite_text
from app.services.postprocess import postprocess

app = FastAPI(title="AI Text Humanizer")

# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


class HumanizeRequest(BaseModel):
    text: str


class HumanizeResponse(BaseModel):
    humanized_text: str
    original_word_count: int
    humanized_word_count: int
    length_ratio: float
    within_limit: bool
    warning: str | None = None


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/humanize", response_model=HumanizeResponse)
async def humanize(request: HumanizeRequest):
    # Step 1: Analyze input
    analysis = analyze_text(request.text)
    
    # Step 2: Get human writing style references via RAG
    style_refs = get_style_references(request.text, n_results=3)
    
    # Step 3: Rewrite with LLM pipeline
    humanized = rewrite_text(request.text, analysis, style_refs)
    
    # Step 4: Post-process and validate
    result = postprocess(request.text, humanized)
    
    return HumanizeResponse(
        humanized_text=result["text"],
        original_word_count=result["original_word_count"],
        humanized_word_count=result["humanized_word_count"],
        length_ratio=result["length_ratio"],
        within_limit=result["within_limit"],
        warning=result["warning"]
    )
```

---

### Frontend (`static/index.html`)

Build a single-page HTML file with:
- A textarea for pasting AI-generated text
- A "Humanize" button
- An output textarea showing the result
- Word count and length ratio display
- Loading spinner during processing
- Copy-to-clipboard button on the output

Keep it clean and simple. Use vanilla HTML/CSS/JS. No frameworks needed.

---

### Dependencies (`requirements.txt`)

```
fastapi>=0.104.0
uvicorn>=0.24.0
openai>=1.6.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
spacy>=3.7.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

Optional (if using Gemini instead of or alongside OpenAI):
```
google-generativeai>=0.3.0
```

---

### Deployment

**Option A: Railway**
```bash
# Procfile
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

**Option B: Render**
- Set build command: `pip install -r requirements.txt`
- Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

**Option C: Docker**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . .
# Build the RAG corpus at build time
RUN python -c "from app.services.rag import build_corpus; build_corpus('app/corpus/samples')"
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Environment Variables

```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...          # Optional, if using Gemini
```

---

## Key Prompt Engineering Techniques That Work

Based on research, these are the techniques that most effectively bypass Originality.ai:

1. **Banned word list** — The single most impactful technique. Eliminates the vocabulary fingerprint detectors look for.

2. **Forced burstiness** — Explicitly telling the LLM to vary sentence lengths between 5 and 25+ words. AI naturally writes uniform 15-20 word sentences.

3. **Contractions and informal connectors** — "don't", "isn't", starting with "But", "And", "So". AI defaults to formal style.

4. **Em dash removal** — Em dashes are one of the strongest AI signals. Replace with commas, periods, or parentheses.

5. **Few-shot style references (RAG)** — Showing the LLM 2-3 examples of real human writing to mimic. This is what the RAG component provides.

6. **Cross-model refinement** — Using a different LLM for the second pass. Research shows Gemini 2.0 Flash produces the most human-like output. So: GPT-4o for initial rewrite, Gemini for refinement (or vice versa).

7. **Higher temperature** — Using temperature 0.8-0.9 increases randomness, making output less predictable (higher perplexity = more human-like to detectors).

8. **Rhetorical fragments and questions** — Real humans use sentence fragments and ask rhetorical questions. AI rarely does this unprompted.

---

## Testing Workflow

Before delivering:

1. Generate 20+ samples of AI text using ChatGPT/Claude on various topics
2. Run each through your humanizer
3. Paste results into Originality.ai
4. Track pass/fail rate — target 80%+
5. If below 80%, iterate on the system prompt:
   - Check which AI patterns are still slipping through
   - Add more specific instructions to the banned list
   - Adjust temperature
   - Try cross-model refinement
   - Add more/better human writing samples to the RAG corpus

---

## Common Pitfalls to Avoid

- **Don't just paraphrase** — Simple synonym replacement doesn't fool modern detectors. You need structural changes.
- **Don't make output too long** — Client requirement is max 30-40% longer. Enforce this in prompts AND post-processing.
- **Don't use the same prompt template every time** — If every output follows the same rewrite pattern, that itself becomes detectable. The RAG component helps add variety.
- **Don't ignore the refinement pass** — The first LLM output will have residual AI patterns. The second pass catches them.
- **Don't rely on tricks like Unicode spaces** — These are fragile, easily patched, and don't produce quality output.

---

## GitHub Repos for Reference

These open-source repos can be studied for implementation patterns:

| Repo | What to learn from it |
|---|---|
| `DadaNanjesha/AI-Text-Humanizer-App` (177 stars) | Streamlit UI structure, spaCy-based analysis |
| `Khizer-Data/AI-Text-Humanizer` | FastAPI + RAG architecture, multiple style modes |
| `Firdavs-coder/ai_humanizer` | Local model via Ollama, smart typo insertion |
| `dixon2004/ai-humanizer` | Gemini API integration, Docker deployment |
| `blader/humanizer` | 24 documented AI writing patterns to detect/remove |
| `CBIhalsen/text-rewriter` | NLP pipeline: POS tagging, synonym replacement |

---

## Summary: Build Order

1. **Set up FastAPI project structure** with the folder layout above
2. **Collect 50-100 human writing samples** and build the ChromaDB corpus
3. **Write the system prompts** (copy from above, then iterate)
4. **Implement the 4-step pipeline**: analyze → RAG retrieve → rewrite → post-process
5. **Build the simple HTML frontend**
6. **Test against Originality.ai** — iterate on prompts until 80%+ pass rate
7. **Deploy** to Railway/Render with a public URL
8. **Hand over the URL to the client**
