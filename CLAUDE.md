# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A web application that takes AI-generated text as input and outputs humanized text that passes Originality.ai detection as "Human" in ≥80% of cases. This is a **custom three-layer NLP pipeline** (T5 paraphrase → algorithmic transforms → LLM grammar polish), not a wrapper around an existing humanizer API. The full technical specification is in `AI_TEXT_HUMANIZER_V2_TECHNICAL_DOC.md`.

## Commands

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (run once)
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger_eng wordnet stopwords

# Download spaCy model (run once)
python -m spacy download en_core_web_sm

# Run the development server
uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Environment Variables

```
OPENAI_API_KEY=sk-...
HF_TOKEN=hf-...
```

## Architecture

The processing pipeline in `app/services/` is a strict 3-layer sequence:

1. **`layer1_paraphrase.py`** — Sentence-tokenizes the input with NLTK, then calls the `Vamsi/T5_Paraphrase_Paws` model via the HF Inference API (one call per sentence). Rejoins sentences into a paragraph.

2. **`layer2_algorithmic.py`** — Five chained deterministic/stochastic NLP transforms:
   - `swap_transitions` — regex-swap transition words within semantic groups
   - `synonym_replace` — WordNet POS-aware synonym substitution (prob=0.20, skip stopwords)
   - `toggle_contractions` — expand all contractions, then randomly re-inject (prob=0.5)
   - `split_long_sentences` — spaCy dep-parse split at `cc`/`mark`/`advcl` for sentences > 20–30 words
   - `merge_short_sentences` — join consecutive short sentences (< 4–8 words) with varied conjunctions

3. **`layer3_polish.py`** — GPT-4o at `temperature=0.0`, grammar-only proofreader. Validated with `difflib.SequenceMatcher`: if similarity ratio < 0.90, falls back to the pre-polish text.

**`rewriter.py`** — Async orchestrator that calls layers 1→2→3 in sequence and computes `burstiness` (np.std of per-sentence word counts) before and after.

**`postprocess.py`** — Validates output length (≤135% of original word count), strips LLM meta-commentary, normalizes whitespace.

The FastAPI app in `app/main.py` exposes `POST /humanize`.

## Key Design Decisions

- **Layer 1 (T5) breaks LLM fingerprints** at the sentence level before any LLM sees the text. This is the highest-impact change vs V1.
- **Layer 2 adds measurable burstiness variance** — synonym swap, contraction toggle, and sentence split/merge all change word counts and rhythm algorithmically, without an LLM re-introducing AI patterns.
- **Layer 3 uses temperature=0** intentionally — the LLM must not be creative, only fix grammar errors. Higher temperature re-introduces AI fingerprints.
- **difflib validation in Layer 3** (ratio ≥ 0.90) is a guardrail: if GPT-4o rewrites too aggressively, the pipeline falls back to the Layer 2 output.
- **Skip flags** (`skip_paraphrase`, `skip_algorithmic`, `skip_polish`) allow ablation testing to isolate which layers contribute most.
- Prompts are stored as Python string constants in `app/prompts/`, not inline in service files.
- `analyzer.py` and `rag.py` are kept in the codebase but are **not called by the V2 pipeline**.

## Project Structure

```
app/
├── main.py                  # FastAPI entry point, /humanize endpoint
├── config.py                # OPENAI_API_KEY, HF_TOKEN, PORT
├── routers/humanize.py      # Route handler, V2 request/response models
├── services/
│   ├── layer1_paraphrase.py # Layer 1: T5 paraphrase via HF Inference API
│   ├── layer2_algorithmic.py# Layer 2: NLP transforms (synonym, contractions, split/merge)
│   ├── layer3_polish.py     # Layer 3: GPT-4o grammar polish + diff validation
│   ├── rewriter.py          # Async orchestrator + burstiness metric
│   ├── postprocess.py       # Length validation, cleanup
│   ├── analyzer.py          # (V1 artifact, kept unused)
│   └── rag.py               # (V1 artifact, kept unused)
├── prompts/
│   ├── layer3_prompt.py     # ANTI_AI_SYSTEM_PROMPT (grammar-only proofreader)
│   ├── system_prompt.py     # (V1 artifact, kept unused)
│   ├── analysis_prompt.py   # (V1 artifact, kept unused)
│   └── refine_prompt.py     # (V1 artifact, kept unused)
└── corpus/                  # (V1 artifact, kept unused)
```

## API

```
POST /humanize

Request:
{
  "text": "...",
  "skip_paraphrase": false,   // optional, default false
  "skip_algorithmic": false,  // optional, default false
  "skip_polish": false        // optional, default false
}

Response:
{
  "original": "...",
  "humanized": "...",
  "layers_applied": ["t5_paraphrase", "algorithmic_nlp", "llm_polish"],
  "burstiness_before": 4.2,
  "burstiness_after": 7.1,
  "original_word_count": 120,
  "humanized_word_count": 134,
  "length_ratio": 1.12,
  "within_limit": true,
  "warning": null
}
```

## Testing

Manual workflow against Originality.ai (no automated test suite defined):
1. Generate 20+ AI text samples via ChatGPT/Claude on varied topics
2. Run through `POST /humanize`
3. Check each on Originality.ai — target ≥80% pass rate
4. Use skip flags to ablate layers and identify which contributes most
5. If below target: tune `synonym_replace` prob, `contract_prob`, or sentence split thresholds
