import re
from typing import List, Set

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="SafeAnswer AI - Grounded QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class Chunk(BaseModel):
    chunk_id: str = Field(..., min_length=1, max_length=100)
    text: str = Field(..., min_length=1, max_length=5000)

    @field_validator("chunk_id", "text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field cannot be blank")
        return v


class QARequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    chunks: List[Chunk] = Field(..., min_length=1)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be blank")
        return v


class QAResponse(BaseModel):
    answer: str
    citations: List[str]
    confidence: float
    answerable: bool


STOPWORDS: Set[str] = {
    "the", "is", "are", "was", "were", "a", "an", "of", "in", "on", "for",
    "to", "and", "or", "what", "when", "where", "who", "which", "how",
    "did", "does", "do", "by", "at", "as", "with", "that", "this", "it",
    "from", "into", "their", "there", "been", "being", "have", "has", "had"
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def tokenize(text: str) -> Set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def unanswerable_response() -> QAResponse:
    return QAResponse(
        answer="I don't know",
        citations=[],
        confidence=0.1,
        answerable=False,
    )


def subject_tokens(question: str) -> Set[str]:
    q_tokens = tokenize(question)
    ignore = {
        "what", "when", "where", "who", "which", "how",
        "release", "released", "open", "opened", "sourced", "launched",
        "developed", "created", "built", "made",
        "written", "programmed", "language",
        "founded", "founder", "year"
    }
    return {t for t in q_tokens if t not in ignore and len(t) > 2}


def question_type(question: str) -> str:
    q = question.lower()
    if "what year" in q or "which year" in q or q.startswith("when "):
        return "year"
    if q.startswith("who ") or " who " in q:
        return "who"
    if any(x in q for x in ["written in", "programmed in", "what language", "which language"]):
        return "language"
    return "unknown"


def exact_chunk_match(question: str, chunk_text: str) -> bool:
    q = question.lower()
    text = chunk_text.lower()
    needed = subject_tokens(question)

    if not needed:
        return False

    chunk_tokens = tokenize(chunk_text)
    if not needed.issubset(chunk_tokens):
        return False

    q_type = question_type(question)

    if q_type == "year":
        if not re.search(r"\b(19|20)\d{2}\b", text):
            return False
        if not any(x in text for x in ["released", "open-sourced", "open sourced", "launched", "developed", "created", "founded"]):
            return False
        return True

    if q_type == "who":
        return any(x in text for x in ["developed by", "created by", "built by", "made by", "founded by"])

    if q_type == "language":
        return any(x in text for x in ["written in", "implemented in", "programmed in"])

    return False


@app.post("/grounded-qa", response_model=QAResponse)
async def grounded_qa(payload: QARequest):
    try:
        question = normalize(payload.question)
        chunks = payload.chunks

        if not question or not chunks:
            return unanswerable_response()

        matches = [chunk for chunk in chunks if exact_chunk_match(question, chunk.text)]

        # Strict abstention policy:
        # zero matches => unanswerable
        # multiple matches => ambiguous => unanswerable
        if len(matches) != 1:
            return unanswerable_response()

        match = matches[0]

        return QAResponse(
            answer=match.text.strip(),
            citations=[match.chunk_id],
            confidence=0.9,
            answerable=True,
        )

    except Exception:
        return unanswerable_response()


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Grounded QA API is running"}