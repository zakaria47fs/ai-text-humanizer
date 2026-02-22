import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routers.humanize import router as humanize_router
from app.config import settings

app = FastAPI(title="AI Text Humanizer")

# Serve static files if the directory exists
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    async def root():
        return FileResponse("static/index.html")

app.include_router(humanize_router)


@app.on_event("startup")
async def startup_event():
    """Build RAG corpus on first startup if chroma_db doesn't exist."""
    if not os.path.exists(settings.CHROMA_DB_PATH):
        from app.services.rag import build_corpus
        samples_dir = settings.CORPUS_SAMPLES_DIR
        if os.path.isdir(samples_dir):
            print(f"Building RAG corpus from {samples_dir}...")
            build_corpus(samples_dir, db_path=settings.CHROMA_DB_PATH)
        else:
            print(f"Corpus samples directory '{samples_dir}' not found. Skipping corpus build.")
