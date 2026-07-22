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


def important_tokens(question: str) -> Set[str]:
    ignore = {
        "what", "when", "where", "who", "which", "how",
        "release", "released", "open", "opened", "sourced", "launched",
        "developed", "created", "built", "made", "written", "programmed",
        "language", "founded", "founder", "year"
    }
    return {t for t in tokenize(question) if t not in ignore and len(t) > 2}


def chunk_matches_question(question: str, chunk_text: str) -> bool:
    q = question.lower()
    text = chunk_text.lower()
    tokens = important_tokens(question)

    if not tokens:
        return False

    if not tokens.issubset(tokenize(chunk_text)):
        return False

    if ("what year" in q or "which year" in q or q.startswith("when ")) and not re.search(r"\b(19|20)\d{2}\b", text):
        return False

    if any(x in q for x in ["written in", "programmed in", "what language", "which language"]) and not any(
        x in text for x in ["written in", "implemented in", "programmed in"]
    ):
        return False

    if (q.startswith("who ") or " who " in q) and not any(
        x in text for x in ["developed by", "created by", "built by", "made by", "founded by"]
    ):
        return False

    return True


@app.post("/grounded-qa", response_model=QAResponse)
async def grounded_qa(payload: QARequest):
    try:
        question = normalize(payload.question)
        chunks = payload.chunks

        if not question or not chunks:
            return unanswerable_response()

        matches = [c for c in chunks if chunk_matches_question(question, c.text)]

        # safest strict mode: exactly one supporting chunk only
        if len(matches) != 1:
            return unanswerable_response()

        matched = matches[0]

        return QAResponse(
            answer=matched.text.strip(),
            citations=[matched.chunk_id],
            confidence=0.9,
            answerable=True,
        )

    except Exception:
        return unanswerable_response()


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Grounded QA API is running"}