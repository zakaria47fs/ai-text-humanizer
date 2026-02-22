# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A web application that takes AI-generated text as input and outputs humanized text that passes Originality.ai detection as "Human" in ≥80% of cases. This is a **custom RAG-based pipeline**, not a wrapper around an existing humanizer API. The full technical specification is in `AI_TEXT_HUMANIZER_TECHNICAL_DOC.md`.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Build the RAG corpus (run once after populating app/corpus/samples/ with .txt files)
python -c "from app.services.rag import build_corpus; build_corpus('app/corpus/samples')"

# Run the development server
uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Environment Variables

```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...   # Optional: for cross-model refinement pass
```

## Architecture

The processing pipeline in `app/services/` is a strict 4-step sequence:

1. **`analyzer.py`** — Detects AI writing patterns in the input: sentence length variance (burstiness), banned AI vocabulary, em dash count, repetitive transition words. Returns a structured `analysis` dict used downstream.

2. **`rag.py`** — Queries ChromaDB (persisted at `./chroma_db/`) for the 3 most stylistically similar human-written samples from `app/corpus/samples/`. Uses `all-MiniLM-L6-v2` (local, free) for embeddings. These samples are injected into the rewrite prompt as few-shot style references.

3. **`rewriter.py`** — Two-pass LLM pipeline:
   - Pass 1: GPT-4o (`temperature=0.85`) with the full system prompt + analysis findings + RAG style references
   - Pass 2: GPT-4o or Gemini (`temperature=0.7`) with the refinement prompt — light cleanup pass only
   - Cross-model refinement (GPT-4o → Gemini) is more effective per the spec

4. **`postprocess.py`** — Validates output length (≤135% of original word count), strips LLM meta-commentary, normalizes whitespace.

The FastAPI app in `app/main.py` exposes `POST /humanize` and serves the frontend from `static/index.html`.

## Key Design Decisions

- **Banned word enforcement is the highest-impact technique** for bypassing Originality.ai. The full banned list lives in `app/prompts/system_prompt.py`; keep it in sync with `analyzer.py`'s `ai_words` list.
- **Em dashes (—) are a strong AI signal** — they are banned in output and detected in analysis.
- **Temperature 0.85** on the first pass is intentional; lower temperature produces more uniform (detectable) output.
- **RAG corpus quality matters**: 50–100 human-written samples of 200–500 words each. Sources: student essays, Wikipedia Good Articles, news excerpts (AP/Reuters). Store as `.txt` in `app/corpus/samples/`.
- The length constraint (≤35% longer than input) is enforced both in the LLM prompt (`CRITICAL RULES` section) and in `postprocess.py`.
- Prompts are stored as Python string constants in `app/prompts/`, not inline in service files.

## Project Structure (to build)

```
app/
├── main.py              # FastAPI entry point, /humanize endpoint
├── config.py            # API keys, settings
├── routers/humanize.py  # Route handler
├── services/
│   ├── analyzer.py      # Step 1: AI pattern detection
│   ├── rag.py           # Step 2: ChromaDB retrieval + corpus builder
│   ├── rewriter.py      # Step 3: Two-pass LLM rewrite
│   └── postprocess.py   # Step 4: Length validation, cleanup
├── prompts/
│   ├── system_prompt.py   # Main humanizer system prompt (banned word list)
│   ├── analysis_prompt.py # AI pattern detection prompt
│   └── refine_prompt.py   # Second-pass refinement prompt
└── corpus/
    ├── build_corpus.py    # One-time corpus builder script
    └── samples/           # Human-written .txt files (50-100, 200-500 words each)
static/index.html          # Single-page frontend (vanilla HTML/CSS/JS)
```

## Testing

Manual workflow against Originality.ai (no automated test suite defined):
1. Generate 20+ AI text samples via ChatGPT/Claude on varied topics
2. Run through `/humanize`
3. Check each on Originality.ai — target ≥80% pass rate
4. If below target: review which AI patterns survive, tighten banned list, adjust temperature, add more corpus samples, or enable cross-model refinement
