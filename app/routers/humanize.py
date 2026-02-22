from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.analyzer import analyze_text
from app.services.rag import get_style_references
from app.services.rewriter import rewrite_text
from app.services.postprocess import postprocess
from app.config import settings

router = APIRouter()


class HumanizeRequest(BaseModel):
    text: str


class HumanizeResponse(BaseModel):
    humanized_text: str
    original_word_count: int
    humanized_word_count: int
    length_ratio: float
    within_limit: bool
    warning: str | None = None


@router.post("/humanize", response_model=HumanizeResponse)
async def humanize(request: HumanizeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Step 1: Analyze input
    analysis = analyze_text(request.text)

    # Step 2: Get human writing style references via RAG
    style_refs = get_style_references(
        request.text,
        n_results=3,
        db_path=settings.CHROMA_DB_PATH
    )

    # Step 3: Rewrite with LLM pipeline
    humanized = rewrite_text(request.text, analysis, style_refs)

    # Step 4: Post-process and validate
    result = postprocess(request.text, humanized)

    return HumanizeResponse(
        humanized_text=result["text"],
        original_word_count=result["original_word_count"],
        humanized_word_count=result["humanized_word_count"],
        length_ratio=result["length_ratio"],
        within_limit=result["within_limit"],
        warning=result["warning"]
    )
