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
    s_tokens = tokenize(sentence)
    return needed.issubset(s_tokens)


def extract_year_answer(question: str, sentence: str) -> Optional[str]:
    if not subject_matches(question, sentence):
        return None
    if not any(x in question.lower() for x in ["what year", "which year", "when "]):
        return None

    year_match = re.search(r"\b(19|20)\d{2}\b", sentence)
    if not year_match:
        return None

    if not any(x in sentence.lower() for x in ["released", "open-sourced", "open sourced", "launched", "developed", "created", "founded"]):
        return None

    return sentence.strip()


def extract_language_answer(question: str, sentence: str) -> Optional[str]:
    q = question.lower()
    s = sentence.lower()

    if not subject_matches(question, sentence):
        return None
    if not any(x in q for x in ["written in", "programmed in", "what language", "which language"]):
        return None
    if not any(x in s for x in ["written in", "implemented in", "programmed in"]):
        return None

    return sentence.strip()


def extract_who_answer(question: str, sentence: str) -> Optional[str]:
    q = question.lower()
    s = sentence.lower()

    if not subject_matches(question, sentence):
        return None
    if not (q.startswith("who ") or " who " in q):
        return None
    if not any(x in s for x in ["developed by", "created by", "built by", "made by", "founded by"]):
        return None

    return sentence.strip()


def extract_generic_answer(question: str, sentence: str) -> Optional[str]:
    if not subject_matches(question, sentence):
        return None

    q_tokens = tokenize(question)
    s_tokens = tokenize(sentence)
    overlap = q_tokens & s_tokens

    if len(overlap) >= max(2, len(q_tokens) // 2):
        return sentence.strip()

    return None


def find_matches(question: str, chunks: List[Chunk]) -> List[Tuple[str, str]]:
    q_type = question_type(question)
    matches = []

    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            answer = None

            if q_type == "year":
                answer = extract_year_answer(question, sentence)
            elif q_type == "language":
                answer = extract_language_answer(question, sentence)
            elif q_type == "who":
                answer = extract_who_answer(question, sentence)
            else:
                answer = extract_generic_answer(question, sentence)

            if answer:
                matches.append((chunk.chunk_id, answer))

    return matches


@app.post("/grounded-qa", response_model=QAResponse)
async def grounded_qa(payload: QARequest):
    try:
        question = normalize(payload.question)
        chunks = payload.chunks

        if not question or not chunks:
            return unanswerable_response()

        matches = find_matches(question, chunks)

        # safest policy: answer only when exactly one strong match exists
        if len(matches) != 1:
            return unanswerable_response()

        chunk_id, answer = matches[0]

        return QAResponse(
            answer=answer,
            citations=[chunk_id],
            confidence=0.9,
            answerable=True,
        )

    except Exception:
        return unanswerable_response()


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Grounded QA API is running"}