import re
from typing import List, Set, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="SafeAnswer AI - Grounded QA API")

# For public testing, keep this simple:
# wildcard origins are okay only when allow_credentials is False.
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


def relation_type(question: str) -> str:
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


def subject_tokens(question: str) -> Set[str]:
    q_tokens = tokenize(question)
    relation_words = {
        "release", "released", "open", "opened", "sourced", "launched",
        "developed", "created", "built", "made",
        "written", "programmed", "language",
        "founded", "founder", "year", "when", "who", "what", "which"
    }
    return {t for t in q_tokens if t not in relation_words and len(t) > 2}


def subject_supported(question: str, sentence: str) -> bool:
    q_subject = subject_tokens(question)
    s_tokens = tokenize(sentence)

    if not q_subject:
        return False

    overlap = q_subject & s_tokens
    return len(overlap) == len(q_subject)


def relation_supported(question: str, sentence: str) -> bool:
    rel = relation_type(question)
    s = sentence.lower()

    if rel == "release":
        return any(x in s for x in ["released", "open-sourced", "open sourced", "launched"])

    if rel == "developed":
        return any(x in s for x in ["developed by", "created by", "built by", "made by", "developed"])

    if rel == "language":
        return any(x in s for x in ["written in", "implemented in", "programmed in"])

    if rel == "founded":
        return any(x in s for x in ["founded by", "founder", "founded"])

    return True


def answer_type_supported(question: str, sentence: str) -> bool:
    q_type = question_type(question)
    s = sentence.lower()

    if q_type == "year":
        return extract_year(sentence) is not None

    if q_type == "who":
        return any(x in s for x in ["developed by", "created by", "built by", "made by", "founded by"])

    if q_type == "language":
        return any(x in s for x in [
            "written in", "implemented in", "programmed in",
            "rust", "python", "java", "javascript", "go", "c++"
        ])

    return True


def sentence_score(question: str, sentence: str) -> float:
    q_tokens = tokenize(question)
    s_tokens = tokenize(sentence)

    if not q_tokens or not s_tokens:
        return 0.0

    overlap = q_tokens & s_tokens
    if not overlap:
        return 0.0

    precision = len(overlap) / len(s_tokens)
    recall = len(overlap) / len(q_tokens)

    if precision + recall == 0:
        return 0.0

    score = (2 * precision * recall) / (precision + recall)

    if question_type(question) == "year" and extract_year(sentence):
        score += 0.15

    return min(score, 1.0)


def exact_sentence_in_chunk(sentence: str, chunk_text: str) -> bool:
    sentences = split_sentences(chunk_text)
    normalized_sentence = normalize(sentence).lower()
    return any(normalize(s).lower() == normalized_sentence for s in sentences)


def find_best_supported_sentence(question: str, chunks: List[Chunk]):
    best_chunk = None
    best_sentence = None
    best_score = 0.0

    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            if not subject_supported(question, sentence):
                continue

            if not relation_supported(question, sentence):
                continue

            if not answer_type_supported(question, sentence):
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

        if not question or not chunks:
            return unanswerable_response()

        best_chunk, best_sentence, best_score = find_best_supported_sentence(question, chunks)

        # Conservative threshold to reduce false positives.
        if best_chunk is None or best_sentence is None or best_score < 0.50:
            return unanswerable_response()

        # The answer must be the exact sentence from the cited chunk.
        if not exact_sentence_in_chunk(best_sentence, best_chunk.text):
            return unanswerable_response()

        return QAResponse(
            answer=best_sentence,
            citations=[best_chunk.chunk_id],
            confidence=0.9,
            answerable=True,
        )

    except Exception:
        return unanswerable_response()


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Grounded QA API is running"}