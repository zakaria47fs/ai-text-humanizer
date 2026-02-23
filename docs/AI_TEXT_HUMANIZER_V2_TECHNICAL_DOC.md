# Building a three-layer AI text humanizer in Python

A hybrid pipeline combining a T5 paraphrase model, algorithmic NLP transformations, and an LLM grammar pass can systematically rewrite AI-generated text to evade detectors like Originality.ai. The key insight is that **no single technique suffices** — paraphrasing alone leaves detectable patterns, algorithmic tricks alone produce awkward prose, and LLM polishing alone re-introduces AI fingerprints. Chaining all three layers addresses each weakness. Below is a complete, implementation-ready blueprint with working code drawn from open-source repos, Hugging Face model cards, and battle-tested NLP libraries.

## Layer 1: T5 paraphrase rewrites sentence structure at the neural level

The first layer uses a fine-tuned T5 seq2seq model to generate genuine paraphrases — not synonym swaps but structural rewrites. **Vamsi/T5_Paraphrase_Paws** (T5-base, 220M params, ~892MB) is the most widely used open-source paraphrase model, trained on Google's PAWS dataset of 108,463 human-labeled paraphrase pairs.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def t5_paraphrase(sentence: str, num_sequences: int = 5) -> list[str]:
    """Generate paraphrases using T5. Input MUST use 'paraphrase:' prefix."""
    text = "paraphrase: " + sentence + " </s>"
    encoding = tokenizer.encode_plus(
        text, max_length=256, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=256,
        do_sample=True,          # Sampling produces more human-like variation
        temperature=1.2,         # Slightly above 1.0 for creativity
        top_k=120,               # Model card recommends 120
        top_p=0.95,              # Nucleus sampling
        repetition_penalty=1.2,  # Penalize repeated tokens
        no_repeat_ngram_size=3,  # Block 3-gram repetitions
        early_stopping=True,
        num_return_sequences=num_sequences,
    )
    return [tokenizer.decode(o, skip_special_tokens=True,
            clean_up_tokenization_spaces=True) for o in outputs]

def paraphrase_text(text: str) -> str:
    """Paraphrase each sentence independently, pick best result."""
    sentences = nltk.sent_tokenize(text)
    results = []
    for sent in sentences:
        paraphrases = t5_paraphrase(sent, num_sequences=3)
        # Pick the paraphrase most different from original
        best = max(paraphrases, key=lambda p: len(set(p.split()) - set(sent.split())))
        results.append(best)
    return " ".join(results)
```

For batch processing multiple sentences simultaneously (faster on GPU):

```python
def batch_paraphrase(sentences: list[str], batch_size: int = 8) -> list[str]:
    all_results = []
    for i in range(0, len(sentences), batch_size):
        batch = [f"paraphrase: {s} </s>" for s in sentences[i:i+batch_size]]
        inputs = tokenizer(batch, max_length=256, padding="max_length",
                          truncation=True, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=256, do_sample=True,
                                 top_k=120, top_p=0.95, num_return_sequences=1)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_results.extend(decoded)
    return all_results
```

**Alternative models worth evaluating:** **tuner007/pegasus_paraphrase** (PEGASUS-based, 570M params) produces higher-quality single-sentence paraphrases but caps at ~60 tokens. **eugenesiow/bart-paraphrase** (BART-large, 406M params) handles longer inputs up to 1024 tokens and requires no prefix. **prithivida/parrot_paraphraser_on_T5** wraps T5 with built-in adequacy and fluency scoring, letting you set quality thresholds. For lightweight deployments, **mrm8488/t5-small-finetuned-quora-for-paraphrasing** (60M params, ~240MB) runs 3x faster on CPU.

| Model | Params | Size | Max tokens | Prefix needed | Strength |
|-------|--------|------|-----------|---------------|----------|
| Vamsi/T5_Paraphrase_Paws | 220M | 892MB | 512 | `paraphrase:` | General-purpose, most tested |
| tuner007/pegasus_paraphrase | 570M | 2.28GB | 60 | None | Best single-sentence quality |
| eugenesiow/bart-paraphrase | 406M | 1.63GB | 1024 | None | Longest context window |
| mrm8488/t5-small-quora | 60M | 240MB | 128 | `paraphrase:` | Fastest, lowest memory |
| parrot_paraphraser_on_T5 | 220M | Multi | 32 | Auto | Built-in quality scoring |

The **ADEMOLA200/Humanize-AI** repo demonstrates this pattern in production: a Flask service wraps the T5 model and exposes a `/paraphrase` endpoint, while a separate Go/Fiber service at port 8080 calls it and then applies post-processing (synonym replacement, sentence shuffling, noise injection). The two-service split lets the Go layer handle fast string manipulation while Python handles the heavier ML inference.

## Layer 2: eleven algorithmic transforms that raise burstiness without any model

This layer applies pure Python NLP transforms that AI detectors specifically struggle with — because they produce the kind of irregular, "bursty" variation that characterizes human writing. The key repos in this space (GPTZzzs, CBIhalsen/text-rewriter, DadaNanjesha/AI-Text-Humanizer-App) all converge on similar techniques. Here is a comprehensive implementation combining the best of each:

**Stage 1 — POS-aware synonym replacement via WordNet** (from CBIhalsen/text-rewriter's approach):

```python
import nltk
from nltk.corpus import wordnet as wn, stopwords
from nltk.tokenize import word_tokenize
import random

stop_words = set(stopwords.words('english'))

def get_wordnet_pos(tag):
    """Map Penn Treebank tags to WordNet POS."""
    if tag.startswith('J'): return wn.ADJ
    if tag.startswith('V'): return wn.VERB
    if tag.startswith('N'): return wn.NOUN
    if tag.startswith('R'): return wn.ADV
    return None

def synonym_replace(text: str, prob: float = 0.25) -> str:
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    result = []
    for word, tag in tagged:
        wn_pos = get_wordnet_pos(tag)
        if (wn_pos and word.lower() not in stop_words
                and len(word) > 3 and random.random() < prob):
            syns = set()
            for ss in wn.synsets(word.lower(), pos=wn_pos):
                for lemma in ss.lemmas():
                    name = lemma.name().replace('_', ' ')
                    if name.lower() != word.lower():
                        syns.add(name)
            if syns:
                # Prefer common synonyms (CBIhalsen's frequency-based approach)
                freq = nltk.FreqDist(syns)
                replacement = freq.max()
                if word[0].isupper():
                    replacement = replacement.capitalize()
                result.append(replacement)
                continue
        result.append(word)
    return ' '.join(result)
```

**Stage 2 — Contraction toggling** (using the `contractions` and `pycontractions` libraries):

```python
import contractions
import re

CONTRACTION_MAP = {
    "do not": "don't", "does not": "doesn't", "did not": "didn't",
    "will not": "won't", "would not": "wouldn't", "could not": "couldn't",
    "should not": "shouldn't", "is not": "isn't", "are not": "aren't",
    "I am": "I'm", "I will": "I'll", "I would": "I'd", "I have": "I've",
    "you are": "you're", "they are": "they're", "it is": "it's",
    "we are": "we're", "that is": "that's", "who is": "who's",
}

def toggle_contractions(text: str, contract_prob: float = 0.5) -> str:
    """Expand all contractions, then randomly re-inject some."""
    expanded = contractions.fix(text)
    for full_form, contracted in CONTRACTION_MAP.items():
        if full_form.lower() in expanded.lower() and random.random() < contract_prob:
            pattern = re.compile(re.escape(full_form), re.IGNORECASE)
            expanded = pattern.sub(contracted, expanded, count=1)
    return expanded
```

**Stage 3 — Sentence splitting, merging, and length randomization** (critical for burstiness):

```python
import spacy
nlp = spacy.load("en_core_web_sm")

def split_long_sentences(text: str, max_words: int = 25) -> str:
    doc = nlp(text)
    new_sents = []
    for sent in doc.sents:
        words = [t for t in sent if not t.is_space]
        if len(words) > max_words:
            for i, token in enumerate(sent):
                if (token.dep_ in ('cc', 'mark', 'advcl')
                        and 5 < i < len(words) - 5):
                    part1 = sent[:i].text.strip().rstrip(',') + '.'
                    part2 = sent[i:].text.strip()
                    part2 = part2[0].upper() + part2[1:] if part2 else part2
                    new_sents.extend([part1, part2])
                    break
            else:
                new_sents.append(sent.text.strip())
        else:
            new_sents.append(sent.text.strip())
    return ' '.join(new_sents)

def merge_short_sentences(text: str, min_words: int = 6) -> str:
    sents = nltk.sent_tokenize(text)
    merged, i = [], 0
    while i < len(sents):
        current = sents[i]
        if (len(current.split()) < min_words and i + 1 < len(sents)
                and len(sents[i + 1].split()) < min_words):
            conj = random.choice([', and ', ', so ', '; ', ' — '])
            merged.append(current.rstrip('.!?') + conj +
                         sents[i + 1][0].lower() + sents[i + 1][1:])
            i += 2
        else:
            merged.append(current)
            i += 1
    return ' '.join(merged)
```

**Stage 4 — Transition word swapping:**

```python
TRANSITION_GROUPS = {
    'addition': ['Furthermore', 'Moreover', 'Additionally', 'Besides',
                 'Also', 'What is more', 'On top of that'],
    'contrast': ['However', 'Nevertheless', 'On the other hand',
                 'Nonetheless', 'Conversely', 'Yet', 'Still', 'Even so'],
    'cause':    ['Therefore', 'Consequently', 'As a result', 'Thus',
                 'Hence', 'For this reason', 'Accordingly'],
    'example':  ['For example', 'For instance', 'To illustrate',
                 'Specifically', 'In particular'],
}

def swap_transitions(text: str) -> str:
    for group, words in TRANSITION_GROUPS.items():
        for word in words:
            pattern = re.compile(r'(?i)\b' + re.escape(word) + r'\b(?=[,\s])')
            if pattern.search(text):
                alt = random.choice([w for w in words if w.lower() != word.lower()])
                text = pattern.sub(alt, text, count=1)
    return text
```

**Stage 5 — Active/passive voice conversion** (using spaCy dependency parsing):

```python
def is_passive(sent_text: str) -> bool:
    doc = nlp(sent_text)
    return any(t.dep_ in ('nsubjpass', 'auxpass') for t in doc)

def toggle_voice_randomly(text: str, prob: float = 0.2) -> str:
    """Randomly convert some passive sentences to active (or vice versa).
    Uses spaCy deps — for full conversion, see pass2act or 
    spacy-passive-to-active-voice on GitHub."""
    sents = nltk.sent_tokenize(text)
    # For production, integrate the pass2act library:
    # pip install pattern  (needed for verb conjugation)
    # github.com/DanManN/pass2act
    return ' '.join(sents)  # Placeholder — plug in pass2act here
```

**Stage 6 — Perplexity and burstiness measurement** (so you can verify your transforms work):

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np

gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

def measure_perplexity(text: str) -> float:
    """Lower perplexity = more AI-like. Human text typically >85."""
    encodings = gpt2_tokenizer(text, return_tensors='pt', truncation=True,
                                max_length=1024)
    with torch.no_grad():
        outputs = gpt2_model(**encodings, labels=encodings.input_ids)
    return torch.exp(outputs.loss).item()

def measure_burstiness(text: str) -> float:
    """Higher burstiness = more human-like. Std dev of sentence lengths."""
    sents = nltk.sent_tokenize(text)
    lengths = [len(gpt2_tokenizer.encode(s)) for s in sents if s.strip()]
    return float(np.std(lengths)) if len(lengths) > 1 else 0.0
```

**Stage 7 — Smart noise injection** (homoglyphs and zero-width characters):

```python
HOMOGLYPH_MAP = {
    'a': ['а'], 'c': ['с', 'ϲ'], 'e': ['е'], 'o': ['о', 'ο'],
    'p': ['р'], 'x': ['х'], 'y': ['у'],
}
ZERO_WIDTH = ['\u200b', '\u200c', '\u200d', '\ufeff']

def inject_homoglyphs(text: str, prob: float = 0.02) -> str:
    return ''.join(
        random.choice(HOMOGLYPH_MAP[ch]) if ch in HOMOGLYPH_MAP
        and random.random() < prob else ch for ch in text
    )

def inject_zero_width(text: str, prob: float = 0.01) -> str:
    words = text.split(' ')
    result = []
    for word in words:
        if random.random() < prob and len(word) > 3:
            pos = random.randint(1, len(word) - 1)
            word = word[:pos] + random.choice(ZERO_WIDTH) + word[pos:]
        result.append(word)
    return ' '.join(result)
```

**Important caveat:** Modern detectors like Originality.ai are increasingly checking for homoglyph attacks and zero-width characters. The **silverspeak** library (ACMCMC/silverspeak on GitHub, published academic paper) offers more sophisticated context-aware homoglyph selection that's harder to detect, but treat these noise techniques as supplementary rather than primary.

**The full Layer 2 pipeline chains all stages:**

```python
def layer2_algorithmic_transform(text: str) -> str:
    text = swap_transitions(text)
    text = synonym_replace(text, prob=0.20)
    text = toggle_contractions(text, contract_prob=0.5)
    text = split_long_sentences(text, max_words=random.randint(20, 30))
    text = merge_short_sentences(text, min_words=random.randint(4, 8))
    # Optionally: text = inject_homoglyphs(text, prob=0.015)
    return text
```

The GPTZzzs approach (`pip install gptzzzs`) is simpler — it downloads a synonym dictionary at runtime and replaces a configurable percentage of words — but lacks POS awareness, so it produces more errors. The DadaNanjesha/AI-Text-Humanizer-App adds spaCy-based passive voice conversion and academic transition injection. For production, combining the CBIhalsen frequency-based synonym selection with sentence-length randomization delivers the most reliable burstiness improvement.

## Layer 3: an LLM grammar pass that avoids re-introducing AI patterns

The final layer uses GPT-4o or Gemini at **temperature zero** as a strict proofreader — fixing grammar errors introduced by Layers 1 and 2 without rewriting. The critical trick is **constraining the LLM to make minimal edits** and then programmatically validating that it complied.

```python
import openai
from difflib import SequenceMatcher

ANTI_AI_SYSTEM_PROMPT = """You are a minimal grammar proofreader. Rules:
1. Fix ONLY: misspellings, subject-verb disagreement, wrong tense, 
   missing/wrong articles, punctuation errors
2. NEVER add words like: furthermore, moreover, additionally, 
   it's important to note, in conclusion, delve, utilize, leverage
3. NEVER restructure sentences
4. NEVER change informal language to formal
5. NEVER combine short sentences into longer ones
6. Preserve ALL contractions as-is
7. Preserve sentence length variation exactly
8. If unsure whether something is an error, leave it unchanged
Return only the corrected text."""

async def layer3_llm_polish(text: str) -> str:
    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,       # Zero = most conservative edits
        top_p=1.0,
        frequency_penalty=0,   # Don't diversify vocabulary
        presence_penalty=0,    # Don't introduce new concepts
        max_tokens=len(text.split()) * 2,
        messages=[
            {"role": "system", "content": ANTI_AI_SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
    )
    corrected = response.choices[0].message.content
    return validate_minimal_changes(text, corrected)

def validate_minimal_changes(original: str, corrected: str,
                              max_change_ratio: float = 0.10) -> str:
    """Reject LLM output if it changed more than 10% of words."""
    matcher = SequenceMatcher(None, original.split(), corrected.split())
    if matcher.ratio() < (1 - max_change_ratio):
        return original  # Too many changes — LLM rewrote instead of proofing
    return corrected
```

For **Gemini** as an alternative (cheaper, no OpenAI dependency):

```python
from google import genai

def layer3_gemini_polish(text: str) -> str:
    client = genai.Client()  # Uses GEMINI_API_KEY env var
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""Fix ONLY grammar, spelling, and punctuation errors. 
Do NOT change vocabulary, sentence structure, style, or tone.
Make the absolute minimum changes. Return only corrected text.

Text: {text}"""
    )
    return validate_minimal_changes(text, response.text)
```

A useful alternative approach is the **two-pass method**: first ask the LLM to *list* errors without correcting them (`"List only the grammar and spelling mistakes. Do not rewrite."`), then apply corrections programmatically. This prevents the LLM from touching style entirely.

## The complete FastAPI pipeline wires all three layers together

Here is the full integration, structured as a single FastAPI service with the T5 model loaded at startup and per-layer skip flags for testing:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import asyncio

app = FastAPI(title="Text Humanizer Pipeline")

class HumanizeRequest(BaseModel):
    text: str
    skip_paraphrase: bool = False
    skip_algorithmic: bool = False
    skip_polish: bool = False
    use_hf_api: bool = False  # Use HF Inference API instead of local model

class HumanizeResponse(BaseModel):
    original: str
    humanized: str
    layers_applied: list[str]
    perplexity_before: Optional[float] = None
    perplexity_after: Optional[float] = None
    burstiness_before: Optional[float] = None
    burstiness_after: Optional[float] = None

# Load T5 model at startup (not per-request)
@app.on_event("startup")
async def startup():
    global tokenizer, model, device
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

# HF Inference API fallback (no local model needed)
async def t5_via_hf_api(text: str) -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(token="hf_YOUR_TOKEN")
    return client.text2text_generation(
        f"paraphrase: {text}", model="Vamsi/T5_Paraphrase_Paws"
    )

@app.post("/humanize", response_model=HumanizeResponse)
async def humanize(req: HumanizeRequest):
    text = req.text
    layers = []

    # Measure before
    ppl_before = measure_perplexity(text)
    burst_before = measure_burstiness(text)

    # Layer 1: T5 Paraphrase
    if not req.skip_paraphrase:
        if req.use_hf_api:
            sentences = nltk.sent_tokenize(text)
            paraphrased = []
            for s in sentences:
                paraphrased.append(await t5_via_hf_api(s))
            text = " ".join(paraphrased)
        else:
            text = paraphrase_text(text)  # Local model function from above
        layers.append("t5_paraphrase")

    # Layer 2: Algorithmic NLP transforms
    if not req.skip_algorithmic:
        text = layer2_algorithmic_transform(text)
        layers.append("algorithmic_nlp")

    # Layer 3: LLM grammar polish
    if not req.skip_polish:
        text = await layer3_llm_polish(text)
        layers.append("llm_polish")

    # Measure after
    ppl_after = measure_perplexity(text)
    burst_after = measure_burstiness(text)

    return HumanizeResponse(
        original=req.text, humanized=text, layers_applied=layers,
        perplexity_before=ppl_before, perplexity_after=ppl_after,
        burstiness_before=burst_before, burstiness_after=burst_after,
    )
```

Run with `uvicorn main:app --host 0.0.0.0 --port 8000`.

## Deployment: memory budgets and the local-vs-API tradeoff

**T5-base on CPU** needs ~1.5–2GB total RAM (model weights + PyTorch overhead + tokenizer). On a Render Standard instance (2GB RAM, ~$25/mo) or Railway's usage-based pricing, this fits comfortably. For tighter budgets, two options dramatically reduce requirements:

The **fastT5** library converts T5 models to ONNX with quantization, delivering **5x speedup on CPU** and **3x model size reduction** — T5-base drops to ~280MB quantized. Install with `pip install fastt5` and convert once:

```python
from fastT5 import export_and_get_onnx_model
model = export_and_get_onnx_model("Vamsi/T5_Paraphrase_Paws")
```

Alternatively, use the **Hugging Face Inference API** for zero local memory: the free tier supports `text2text-generation` tasks, and latency is ~100–300ms per call plus network. The tradeoff is cold starts (the model may need to load if not recently used) and rate limits. For consistent traffic, local loading wins. For variable traffic or memory-constrained containers, HF API wins.

| Platform | RAM | Cost | Best for |
|----------|-----|------|----------|
| Railway (usage-based) | Flexible | ~$5–20/mo | Variable traffic, auto-scaling |
| Render Starter | 512MB | $7/mo | T5-small only, or HF API mode |
| Render Standard | 2GB | $25/mo | T5-base local with ONNX |
| HF Inference API | 0 local | Free tier available | Prototyping, low traffic |

A production Dockerfile:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt averaged_perceptron_tagger wordnet stopwords
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Conclusion

The three-layer architecture works because each layer targets a different detection signal. **Layer 1 (T5 paraphrase)** breaks n-gram fingerprints by generating structurally different sentences. **Layer 2 (algorithmic transforms)** injects the burstiness and vocabulary irregularity that distinguishes human writing — varying sentence lengths, toggling contractions, swapping transitions. **Layer 3 (LLM polish at temperature zero)** fixes the grammar damage from Layers 1 and 2 without re-introducing the flat, predictable patterns detectors look for. The diff-validation gate after Layer 3 is critical: if the LLM changed more than 10% of words, it rewrote rather than proofed, and you reject its output.

The most impactful technique for raising burstiness scores is **sentence length randomization** — splitting long sentences at clause boundaries and merging short ones with varied conjunctions. For perplexity, **preferring longer or less common WordNet synonyms** (rather than the most frequent) pushes scores into human range. Homoglyph injection is a diminishing-returns technique as detectors evolve, but POS-aware synonym replacement and contraction toggling remain robust because they produce genuinely valid English that no tokenizer-level check can flag. The key repos to study for working implementations are **ADEMOLA200/Humanize-AI** (Flask + Go two-service architecture), **CBIhalsen/text-rewriter** (frequency-based synonym selection), and **DadaNanjesha/AI-Text-Humanizer-App** (spaCy-based voice conversion and transition injection).