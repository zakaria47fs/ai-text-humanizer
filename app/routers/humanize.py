import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.rewriter import rewrite_text
from app.services.postprocess import postprocess

router = APIRouter()
logger = logging.getLogger(__name__)


class HumanizeRequest(BaseModel):
    text: str
    skip_paraphrase: bool = False
    skip_algorithmic: bool = False
    skip_polish: bool = False


class HumanizeResponse(BaseModel):
    original: str
    humanized: str
    layers_applied: list[str]
    burstiness_before: float | None
    burstiness_after: float | None
    original_word_count: int
    humanized_word_count: int
    length_ratio: float
    within_limit: bool
    warning: str | None = None


@router.post("/humanize", response_model=HumanizeResponse)
async def humanize(request: HumanizeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    word_count = len(request.text.split())
    logger.info("Request received | words=%d | skip_paraphrase=%s skip_algorithmic=%s skip_polish=%s",
                word_count, request.skip_paraphrase, request.skip_algorithmic, request.skip_polish)

    try:
        result = await rewrite_text(
            request.text,
            skip_paraphrase=request.skip_paraphrase,
            skip_algorithmic=request.skip_algorithmic,
            skip_polish=request.skip_polish,
        )
    except Exception:
        logger.exception("Pipeline failed")
        raise

    post = postprocess(request.text, result["text"])

    logger.info("Request done   | layers=%s | burstiness %.2f → %.2f | words %d → %d | within_limit=%s",
                result["layers"],
                result["burst_before"] or 0,
                result["burst_after"] or 0,
                post["original_word_count"],
                post["humanized_word_count"],
                post["within_limit"])

    return HumanizeResponse(
        original=request.text,
        humanized=post["text"],
        layers_applied=result["layers"],
        burstiness_before=result["burst_before"],
        burstiness_after=result["burst_after"],
        original_word_count=post["original_word_count"],
        humanized_word_count=post["humanized_word_count"],
        length_ratio=post["length_ratio"],
        within_limit=post["within_limit"],
        warning=post["warning"],
    )
