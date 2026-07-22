import re
from typing import List, Optional, Set, Tuple

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


def split_sentences(text: str) -> List[str]:
    text = normalize(text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def tokenize(text: str) -> Set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def extract_year(text: str) -> Optional[str]:
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return m.group(0) if m else None


def unanswerable_response() -> QAResponse:
    return QAResponse(
        answer="I don't know",
        citations=[],
        confidence=0.1,
        answerable=False,
    )


def question_type(question: str) -> str:
    q = question.lower()
    if "what year" in q or "which year" in q or q.startswith("when "):
        return "year"
    if q.startswith("who ") or " who " in q:
        return "who"
    if any(x in q for x in ["written in", "programmed in", "what language", "which language"]):
        return "language"
    return "generic"


def subject_tokens(question: str) -> Set[str]:
    q_tokens = tokenize(question)
    ignore = {
        "release", "released", "open", "opened", "sourced", "launched",
        "developed", "created", "built", "made",
        "written", "programmed", "language",
        "founded", "founder", "year", "when", "who", "what", "which"
    }
    return {t for t in q_tokens if t not in ignore and len(t) > 2}


def subject_matches(question: str, sentence: str) -> bool:
    needed = subject_tokens(question)
    if not needed:
        return False
    return needed.issubset(tokenize(sentence))


def year_sentence_matches(question: str, sentence: str) -> bool:
    q = question.lower()
    s = sentence.lower()
    return (
        subject_matches(question, sentence)
        and any(x in q for x in ["what year", "which year", "when "])
        and extract_year(sentence) is not None
        and any(x in s for x in ["released", "open-sourced", "open sourced", "launched", "developed", "created", "founded"])
    )


def who_sentence_matches(question: str, sentence: str) -> bool:
    q = question.lower()
    s = sentence.lower()
    return (
        subject_matches(question, sentence)
        and (q.startswith("who ") or " who " in q)
        and any(x in s for x in ["developed by", "created by", "built by", "made by", "founded by"])
    )


def language_sentence_matches(question: str, sentence: str) -> bool:
    q = question.lower()
    s = sentence.lower()
    return (
        subject_matches(question, sentence)
        and any(x in q for x in ["written in", "programmed in", "what language", "which language"])
        and any(x in s for x in ["written in", "implemented in", "programmed in"])
    )


def generic_sentence_matches(question: str, sentence: str) -> bool:
    if not subject_matches(question, sentence):
        return False
    q_tokens = tokenize(question)
    s_tokens = tokenize(sentence)
    overlap = q_tokens & s_tokens
    return len(overlap) >= max(2, len(q_tokens) // 2)


def sentence_matches(question: str, sentence: str) -> bool:
    q_type = question_type(question)
    if q_type == "year":
        return year_sentence_matches(question, sentence)
    if q_type == "who":
        return who_sentence_matches(question, sentence)
    if q_type == "language":
        return language_sentence_matches(question, sentence)
    return generic_sentence_matches(question, sentence)


def find_exact_matches(question: str, chunks: List[Chunk]) -> List[Tuple[str, str]]:
    matches = []
    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            if sentence_matches(question, sentence):
                matches.append((chunk.chunk_id, sentence))
    return matches


@app.post("/grounded-qa", response_model=QAResponse)
async def grounded_qa(payload: QARequest):
    try:
        question = normalize(payload.question)
        chunks = payload.chunks

        if not question or not chunks:
            return unanswerable_response()

        matches = find_exact_matches(question, chunks)

        # No exact grounded support
        if len(matches) == 0:
            return unanswerable_response()

        # More than one support sentence => ambiguous, safest is abstain
        if len(matches) > 1:
            return unanswerable_response()

        chunk_id, exact_sentence = matches[0]

        return QAResponse(
            answer=exact_sentence,
            citations=[chunk_id],
            confidence=0.9,
            answerable=True,
        )

    except Exception:
        return unanswerable_response()


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Grounded QA API is running"}