import re
from typing import List, Set, Optional

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
    return "unknown"


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


def subject_matches(question: str, sentence: str) -> bool:
    needed = subject_tokens(question)
    if not needed:
        return False
    return needed.issubset(tokenize(sentence))


def extract_candidate_answer(question: str, sentence: str) -> Optional[str]:
    q_type = question_type(question)
    s = sentence.lower()

    if not subject_matches(question, sentence):
        return None

    if q_type == "year":
        if extract_year(sentence) and any(x in s for x in [
            "released", "open-sourced", "open sourced", "launched",
            "developed", "created", "founded"
        ]):
            return sentence.strip()

    if q_type == "who":
        if any(x in s for x in [
            "developed by", "created by", "built by", "made by", "founded by"
        ]):
            return sentence.strip()

    if q_type == "language":
        if any(x in s for x in [
            "written in", "implemented in", "programmed in"
        ]):
            return sentence.strip()

    return None


def find_candidate_answer(question: str, chunks: List[Chunk]) -> Optional[str]:
    candidates = []

    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            ans = extract_candidate_answer(question, sentence)
            if ans:
                candidates.append(ans)

    unique_candidates = list(dict.fromkeys(candidates))

    if len(unique_candidates) != 1:
        return None

    return unique_candidates[0]


def find_supporting_chunk_ids(answer: str, chunks: List[Chunk]) -> List[str]:
    answer_norm = normalize(answer).lower()
    supporting = []

    for chunk in chunks:
        chunk_text_norm = normalize(chunk.text).lower()
        if answer_norm in chunk_text_norm:
            supporting.append(chunk.chunk_id)

    return supporting


@app.post("/grounded-qa", response_model=QAResponse)
async def grounded_qa(payload: QARequest):
    try:
        question = normalize(payload.question)
        chunks = payload.chunks

        if not question or not chunks:
            return unanswerable_response()

        candidate_answer = find_candidate_answer(question, chunks)
        if not candidate_answer:
            return unanswerable_response()

        supporting_chunk_ids = find_supporting_chunk_ids(candidate_answer, chunks)

        # Deterministic citation repair:
        # cite only if exactly one chunk literally contains the exact answer text
        if len(supporting_chunk_ids) != 1:
            return unanswerable_response()

        return QAResponse(
            answer=candidate_answer,
            citations=[supporting_chunk_ids[0]],
            confidence=0.9,
            answerable=True,
        )

    except Exception:
        return unanswerable_response()


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Grounded QA API is running"}