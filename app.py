import re
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="SafeAnswer AI - Grounded QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Chunk(BaseModel):
    chunk_id: Optional[str] = None
    text: Optional[str] = None

class QARequest(BaseModel):
    question: Optional[str] = None
    chunks: Optional[List[Chunk]] = None

STOPWORDS = {
    "the", "is", "was", "are", "were", "a", "an", "of", "in", "on", "for",
    "to", "and", "or", "what", "when", "where", "who", "which", "how",
    "did", "does", "do", "by", "at", "as", "with", "that", "this", "it"
}

def tokenize(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS}

def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

def unanswerable_response():
    return {
        "answer": "I don't know",
        "citations": [],
        "confidence": 0.1,
        "answerable": False,
    }

def extract_year(text: str):
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return m.group(0) if m else None

@app.post("/grounded-qa")
async def grounded_qa(payload: QARequest):
    try:
        question = (payload.question or "").strip()
        chunks = payload.chunks or []

        if not question or not chunks:
            return unanswerable_response()

        q_tokens = tokenize(question)
        if not q_tokens:
            return unanswerable_response()

        best_chunk = None
        best_chunk_score = 0.0

        for chunk in chunks:
            if not chunk.chunk_id or not chunk.text:
                continue
            c_tokens = tokenize(chunk.text)
            if not c_tokens:
                continue
            overlap = q_tokens & c_tokens
            score = len(overlap) / max(len(q_tokens), 1)
            if score > best_chunk_score:
                best_chunk_score = score
                best_chunk = chunk

        if best_chunk is None or best_chunk_score < 0.5:
            return unanswerable_response()

        best_sentence = None
        best_sentence_score = 0.0

        for sentence in split_sentences(best_chunk.text):
            s_tokens = tokenize(sentence)
            if not s_tokens:
                continue
            overlap = q_tokens & s_tokens
            score = len(overlap) / max(len(q_tokens), 1)
            if score > best_sentence_score:
                best_sentence_score = score
                best_sentence = sentence

        if not best_sentence or best_sentence_score < 0.5:
            return unanswerable_response()

        q_lower = question.lower()

        if "what year" in q_lower or "which year" in q_lower:
            year = extract_year(best_sentence)
            if not year:
                return unanswerable_response()
            answer = f"{year}"
        else:
            answer = best_sentence

        confidence = round(min(0.95, 0.55 + best_sentence_score * 0.4), 2)

        return {
            "answer": answer,
            "citations": [best_chunk.chunk_id],
            "confidence": confidence,
            "answerable": True,
        }

    except Exception:
        return unanswerable_response()

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Grounded QA API is running"}