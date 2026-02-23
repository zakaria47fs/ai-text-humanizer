import difflib
import os

import openai

_client = None


def _get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def validate_minimal_changes(original: str, corrected: str, max_change_ratio: float = 0.10) -> bool:
    """Return True if corrected is sufficiently similar to original (ratio >= 0.90)."""
    ratio = difflib.SequenceMatcher(None, original, corrected).ratio()
    return ratio >= (1.0 - max_change_ratio)


async def layer3_polish(text: str) -> str:
    """GPT-4o grammar proofreader at temperature=0."""
    from app.prompts.layer3_prompt import ANTI_AI_SYSTEM_PROMPT

    client = _get_client()
    response = await client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        messages=[
            {"role": "system", "content": ANTI_AI_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    corrected = response.choices[0].message.content.strip()

    if not validate_minimal_changes(text, corrected):
        # LLM made too many changes; fall back to the pre-polish text
        return text
    return corrected
