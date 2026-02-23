import os

import openai
from langsmith import traceable
from langsmith.wrappers import wrap_openai

from app.prompts.system_prompt import SYSTEM_PROMPT
from app.prompts.refine_prompt import REFINE_PROMPT

_client = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = wrap_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    return _client


@traceable(name="pass1-rewrite", tags=["humanizer"])
def _pass1(client, system_prompt: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.85,
        top_p=0.9,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )
    return response.choices[0].message.content


@traceable(name="pass2-refine", tags=["humanizer"])
def _pass2(client, refine_prompt: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[
            {"role": "system", "content": refine_prompt},
            {"role": "user", "content": user_message},
        ]
    )
    return response.choices[0].message.content


@traceable(name="rewrite-pipeline", tags=["humanizer"])
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

    pass1_text = _pass1(
        client,
        system_prompt=SYSTEM_PROMPT,
        user_message=f"""Rewrite the following AI-generated text to sound naturally human-written.

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
- Do NOT add any meta-commentary about the rewriting process"""
    )

    return _pass2(
        client,
        refine_prompt=REFINE_PROMPT,
        user_message=f"""Review and lightly refine this text. Fix any remaining AI-like patterns while keeping the natural human voice. Do NOT make it longer.

Maximum word count: {max_words} words

TEXT TO REFINE:
{pass1_text}"""
    )
