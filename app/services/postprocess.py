def postprocess(original_text: str, humanized_text: str, max_ratio: float = 1.35) -> dict:
    """Validate output meets length requirements and clean up."""

    original_words = len(original_text.split())
    humanized_words = len(humanized_text.split())
    ratio = humanized_words / original_words if original_words > 0 else 1.0

    # Clean up any extra whitespace
    humanized_text = " ".join(humanized_text.split())

    # Remove any accidental meta-commentary the LLM might have added
    lines = humanized_text.split("\n")
    cleaned_lines = [
        line for line in lines
        if not line.strip().lower().startswith(("here is", "here's the", "note:", "rewritten", "humanized"))
    ]
    humanized_text = "\n".join(cleaned_lines).strip()

    # Recount after cleanup
    humanized_words = len(humanized_text.split())
    ratio = humanized_words / original_words if original_words > 0 else 1.0

    return {
        "text": humanized_text,
        "original_word_count": original_words,
        "humanized_word_count": humanized_words,
        "length_ratio": round(ratio, 2),
        "within_limit": ratio <= max_ratio,
        "warning": f"Text is {round((ratio - 1) * 100)}% longer than original" if ratio > max_ratio else None
    }
