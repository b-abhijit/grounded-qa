import re
from typing import List, Set

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="SafeAnswer AI - Grounded QA API")

# Safer CORS:
# If you want allow_credentials=True, use explicit origins.
# For assignment/demo use, easiest is credentials=False with wildcard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
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


def extract_year(text: str) -> str | None:
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return m.group(0) if m else None


def unanswerable_response() -> QAResponse:
    return QAResponse(
        answer="I don't know",
        citations=[],
        confidence=0.1,
        answerable=False,
    )


def detect_question_type(question: str) -> str:
    q = question.lower()
    if "what year" in q or "which year" in q or q.startswith("when "):
        return "year"
    if q.startswith("who ") or " who " in q:
        return "who"
    if "written in" in q or "programmed in" in q or "what language" in q or "which language" in q:
        return "language"
    return "generic"


def detect_relation_type(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ["release", "released", "open-sourced", "open sourced", "launched"]):
        return "release"
    if any(x in q for x in ["developed", "created", "built", "made"]):
        return "developed"
    if any(x in q for x in ["written in", "programmed in", "language"]):
        return "language"
    if any(x in q for x in ["founded", "founder"]):
        return "founded"
    return "generic"


def question_type_supported(q_type: str, sentence: str) -> bool:
    s = sentence.lower()
    if q_type == "year":
        return extract_year(sentence) is not None
    if q_type == "who":
        return any(x in s for x in ["developed by", "created by", "built by", "made by", "founded by"])
    if q_type == "language":
        return any(x in s for x in ["written in", "implemented in", "programmed in", "rust", "python", "java", "javascript", "go", "c++"])
    return True


def relation_supported(rel_type: str, sentence: str) -> bool:
    s = sentence.lower()
    if rel_type == "release":
        return any(x in s for x in ["released", "open-sourced", "open sourced", "launched"])
    if rel_type == "developed":
        return any(x in s for x in ["developed by", "created by", "built by", "made by", "developed"])
    if rel_type == "language":
        return any(x in s for x in ["written in", "implemented in", "programmed in"])
    if rel_type == "founded":
        return any(x in s for x in ["founded by", "founder", "founded"])
    return True


def subject_overlap(question: str, sentence: str) -> bool:
    q_tokens = tokenize(question)
    s_tokens = tokenize(sentence)

    relation_words = {
        "release", "released", "open", "opened", "open", "sourced", "launched",
        "developed", "created", "built", "made",
        "written", "programmed", "language",
        "founded", "founder", "year", "when", "who", "what", "which"
    }

    entity_tokens = {t for t in q_tokens if t not in relation_words and len(t) > 2}
    if not entity_tokens:
        return False

    overlap = entity_tokens & s_tokens
    return len(overlap) >= max(1, len(entity_tokens) - 1)


def sentence_score(question: str, sentence: str) -> float:
    q_tokens = tokenize(question)
    s_tokens = tokenize(sentence)
    if not q_tokens or not s_tokens:
        return 0.0

    overlap = q_tokens & s_tokens
    precision = len(overlap) / len(s_tokens)
    recall = len(overlap) / len(q_tokens)

    if precision + recall == 0:
        return 0.0

    score = (2 * precision * recall) / (precision + recall)

    if extract_year(sentence) and detect_question_type(question) == "year":
        score += 0.15

    return min(score, 1.0)


def exact_support_check(answer: str, chunk_text: str) -> bool:
    return normalize(answer).lower() in normalize(chunk_text).lower()


def find_best_support(question: str, chunks: List[Chunk]):
    best_chunk = None
    best_sentence = None
    best_score = 0.0

    q_type = detect_question_type(question)
    rel_type = detect_relation_type(question)

    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            if not subject_overlap(question, sentence):
                continue
            if not relation_supported(rel_type, sentence):
                continue
            if not question_type_supported(q_type, sentence):
                continue

            score = sentence_score(question, sentence)
            if score > best_score:
                best_score = score
                best_chunk = chunk
                best_sentence = sentence

    return best_chunk, best_sentence, best_score


@app.post("/grounded-qa", response_model=QAResponse)
async def grounded_qa(payload: QARequest):
    try:
        question = normalize(payload.question)
        chunks = payload.chunks

        best_chunk, best_sentence, best_score = find_best_support(question, chunks)

        if best_chunk is None or best_sentence is None or best_score < 0.35:
            return unanswerable_response()

        answer = best_sentence.strip()

        if not exact_support_check(answer, best_chunk.text):
            return unanswerable_response()

        confidence = round(min(0.95, 0.45 + best_score * 0.45), 2)

        return QAResponse(
            answer=answer,
            citations=[best_chunk.chunk_id],
            confidence=confidence,
            answerable=True,
        )

    except Exception:
        return unanswerable_response()


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Grounded QA API is running"}