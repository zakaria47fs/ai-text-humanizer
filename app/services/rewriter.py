import numpy as np
import nltk

from app.services.layer1_paraphrase import layer1_paraphrase
from app.services.layer2_algorithmic import layer2_transform
from app.services.layer3_polish import layer3_polish

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def measure_burstiness(text: str) -> float | None:
    """Standard deviation of per-sentence word counts."""
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2:
        return None
    lengths = [len(s.split()) for s in sentences]
    return float(np.std(lengths))


async def rewrite_text(
    text: str,
    skip_paraphrase: bool = False,
    skip_algorithmic: bool = False,
    skip_polish: bool = False,
) -> dict:
    """Three-layer humanization pipeline."""
    layers = []
    burst_before = measure_burstiness(text)

    if not skip_paraphrase:
        text = await layer1_paraphrase(text)
        layers.append("t5_paraphrase")

    if not skip_algorithmic:
        text = layer2_transform(text)
        layers.append("algorithmic_nlp")

    if not skip_polish:
        text = await layer3_polish(text)
        layers.append("llm_polish")

    burst_after = measure_burstiness(text)

    return {
        "text": text,
        "layers": layers,
        "burst_before": burst_before,
        "burst_after": burst_after,
    }
