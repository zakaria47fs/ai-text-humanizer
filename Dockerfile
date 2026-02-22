FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the RAG corpus at image build time (requires samples to exist)
RUN python -c "from app.services.rag import build_corpus; build_corpus('app/corpus/samples')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
