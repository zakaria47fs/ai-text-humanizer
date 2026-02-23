import random
import re

import nltk
import spacy
import contractions
from nltk.corpus import wordnet, stopwords

nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


TRANSITION_GROUPS = [
    ["furthermore", "additionally", "moreover", "in addition", "also"],
    ["however", "nevertheless", "that said", "even so", "on the other hand"],
    ["therefore", "thus", "as a result", "consequently", "hence"],
    ["for example", "for instance", "to illustrate", "specifically", "namely"],
    ["in conclusion", "to summarize", "overall", "in summary", "all in all"],
    ["first", "to begin with", "initially", "first of all"],
    ["finally", "lastly", "in the end", "ultimately"],
]


def _build_transition_map() -> dict:
    mapping = {}
    for group in TRANSITION_GROUPS:
        for word in group:
            alternatives = [w for w in group if w != word]
            mapping[word.lower()] = alternatives
    return mapping


_TRANSITION_MAP = _build_transition_map()

# Sorted longest-first to prevent partial matches
_TRANSITION_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(t) for t in sorted(_TRANSITION_MAP.keys(), key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)


def swap_transitions(text: str) -> str:
    """Swap transition words with alternatives from the same semantic group."""
    def replace_match(m: re.Match) -> str:
        phrase = m.group(0)
        lower = phrase.lower()
        alternatives = _TRANSITION_MAP.get(lower)
        if not alternatives:
            return phrase
        replacement = random.choice(alternatives)
        if phrase[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        return replacement

    return _TRANSITION_PATTERN.sub(replace_match, text)


def _get_wordnet_pos(treebank_tag: str):
    """Map NLTK POS tag to WordNet POS constant."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return None


def synonym_replace(text: str, prob: float = 0.20) -> str:
    """Replace non-stopword content words with WordNet synonyms (POS-aware)."""
    stop_words = set(stopwords.words("english"))
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    result = []

    for word, pos in pos_tags:
        wn_pos = _get_wordnet_pos(pos)
        if (
            random.random() < prob
            and wn_pos is not None
            and word.lower() not in stop_words
            and word.isalpha()
            and len(word) > 3
        ):
            synsets = wordnet.synsets(word, pos=wn_pos)
            candidates = []
            for syn in synsets:
                for lemma in syn.lemmas():
                    candidate = lemma.name().replace("_", " ")
                    if candidate.lower() != word.lower() and "_" not in lemma.name():
                        candidates.append((candidate, lemma.count()))

            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                chosen, _ = random.choice(candidates[:3])
                if word[0].isupper():
                    chosen = chosen[0].upper() + chosen[1:]
                result.append(chosen)
                continue

        result.append(word)

    # Reconstruct text preserving punctuation spacing
    output = ""
    punct_attach = {".", ",", "!", "?", ";", ":", "'s", "n't", "'re", "'ve", "'ll", "'d", "'m"}
    open_brackets = {"(", "[", "{"}
    close_brackets = {")", "]", "}"}

    for i, token in enumerate(result):
        if i == 0:
            output = token
        elif token in punct_attach or (output and output[-1] in open_brackets) or token in close_brackets:
            output += token
        else:
            output += " " + token

    return output


def toggle_contractions(text: str, contract_prob: float = 0.5) -> str:
    """Expand all contractions then randomly re-inject a subset."""
    expanded = contractions.fix(text)

    contraction_map = {
        "do not": "don't",
        "does not": "doesn't",
        "did not": "didn't",
        "can not": "can't",
        "cannot": "can't",
        "will not": "won't",
        "would not": "wouldn't",
        "should not": "shouldn't",
        "could not": "couldn't",
        "is not": "isn't",
        "are not": "aren't",
        "was not": "wasn't",
        "were not": "weren't",
        "have not": "haven't",
        "has not": "hasn't",
        "had not": "hadn't",
        "I am": "I'm",
        "I will": "I'll",
        "I have": "I've",
        "I would": "I'd",
        "you are": "you're",
        "you will": "you'll",
        "you have": "you've",
        "you would": "you'd",
        "they are": "they're",
        "they will": "they'll",
        "they have": "they've",
        "we are": "we're",
        "we will": "we'll",
        "we have": "we've",
        "it is": "it's",
        "that is": "that's",
        "there is": "there's",
        "here is": "here's",
    }

    result = expanded
    for expanded_form, contracted_form in contraction_map.items():
        if random.random() < contract_prob:
            pattern = re.compile(re.escape(expanded_form), re.IGNORECASE)

            def make_replacer(contraction: str):
                def replacer(m: re.Match) -> str:
                    if m.group(0)[0].isupper():
                        return contraction[0].upper() + contraction[1:]
                    return contraction
                return replacer

            result = pattern.sub(make_replacer(contracted_form), result)

    return result


def split_long_sentences(text: str, max_words: int = None) -> str:
    """Split long sentences at clause boundaries using spaCy dependency parse."""
    if max_words is None:
        max_words = random.randint(20, 30)

    nlp = _get_nlp()
    doc = nlp(text)
    result_sentences = []

    for sent in doc.sents:
        words = [t for t in sent if not t.is_space]
        if len(words) <= max_words:
            result_sentences.append(sent.text.strip())
            continue

        split_tokens = [
            t for t in sent
            if t.dep_ in {"cc", "mark", "advcl"} and t.i > sent.start + 2
        ]

        if not split_tokens:
            result_sentences.append(sent.text.strip())
            continue

        split_tok = split_tokens[0]
        sent_start = sent.start_char
        left = sent.text[: split_tok.idx - sent_start].strip().rstrip(",")
        right = sent.text[split_tok.idx - sent_start :].strip()

        if left and right:
            result_sentences.append(left + ".")
            result_sentences.append(right[0].upper() + right[1:])
        else:
            result_sentences.append(sent.text.strip())

    return " ".join(result_sentences)


MERGE_CONJUNCTIONS = ["and", "but", "so", "yet", "while", "although", "because", "since"]


def merge_short_sentences(text: str, min_words: int = None) -> str:
    """Merge consecutive short sentences with varied conjunctions."""
    if min_words is None:
        min_words = random.randint(4, 8)

    nlp = _get_nlp()
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents]

    result = []
    i = 0
    while i < len(sentences):
        word_count = len(sentences[i].split())
        if word_count < min_words and i + 1 < len(sentences):
            conj = random.choice(MERGE_CONJUNCTIONS)
            left = sentences[i].rstrip(".")
            next_sent = sentences[i + 1]
            merged = f"{left}, {conj} {next_sent[0].lower()}{next_sent[1:]}"
            result.append(merged)
            i += 2
        else:
            result.append(sentences[i])
            i += 1

    return " ".join(result)


def layer2_transform(text: str) -> str:
    """Apply all algorithmic transforms in sequence."""
    text = swap_transitions(text)
    text = synonym_replace(text, prob=0.20)
    text = toggle_contractions(text, contract_prob=0.5)
    text = split_long_sentences(text)
    text = merge_short_sentences(text)
    return text
