import os

import openai

from app.prompts.system_prompt import SYSTEM_PROMPT
from app.prompts.refine_prompt import REFINE_PROMPT

_client = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def rewrite_text(
    original_text: str,
    analysis: dict,
    style_references: list[str]
) -> str:
    """Main two-pass rewriting pipeline."""

    client = _get_client()

    # Format style references for the prompt
    if style_references:
        style_block = "\n\n---\n\n".join(
            [f"STYLE REFERENCE {i+1}:\n{ref}" for i, ref in enumerate(style_references)]
        )
    else:
        style_block = "No style references available."

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

    max_words = int(analysis["original_word_count"] * 1.35)

    # --- PASS 1: Main rewrite ---
    pass1_response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.85,
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
- Maximum output: {max_words} words
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

Maximum word count: {max_words} words

TEXT TO REFINE:
{pass1_text}"""}
        ]
    )

    return pass2_response.choices[0].message.content
