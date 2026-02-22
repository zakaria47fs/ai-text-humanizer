import re
from collections import Counter


def analyze_text(text: str) -> dict:
    """Analyze text for AI writing patterns. Returns a report dict."""

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentence_lengths = [len(s.split()) for s in sentences]

    # Burstiness: variance in sentence lengths
    avg_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    variance = sum((l - avg_len) ** 2 for l in sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
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
    em_dash_count = text.count("\u2014") + text.count("--")

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
