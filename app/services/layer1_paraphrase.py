import os

import httpx
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

_HF_API_URL = "https://api-inference.huggingface.co/models/Vamsi/T5_Paraphrase_Paws"


async def layer1_paraphrase(text: str) -> str:
    """Paraphrase text sentence-by-sentence via HF Inference API (httpx)."""
    token = os.getenv("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"}
    sentences = nltk.sent_tokenize(text)
    paraphrased = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for s in sentences:
            response = await client.post(
                _HF_API_URL,
                headers=headers,
                json={"inputs": f"paraphrase: {s} </s>"},
            )
            if response.status_code == 200:
                data = response.json()
                # T5 returns [{"generated_text": "..."}]
                if isinstance(data, list) and data:
                    paraphrased.append(data[0].get("generated_text", s).strip())
                else:
                    paraphrased.append(s)
            else:
                # Fall back to original sentence on API error
                paraphrased.append(s)

    return " ".join(paraphrased)
