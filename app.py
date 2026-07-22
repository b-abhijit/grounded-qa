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
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

class Chunk(BaseModel):
    chunk_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)

    @field_validator("chunk_id", "text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field cannot be blank")
        return v

class QARequest(BaseModel):
    question: str = Field(..., min_length=1)
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

def unanswerable_response():
    return QAResponse(answer="I don't know", citations=[], confidence=0.1, answerable=False)

def question_type(question: str) -> str:
    q = question.lower()
    if "what year" in q or "which year" in q or q.startswith("when "):
        return "year"
    if q.startswith("who ") or " who " in q:
        return "who"
    if "written in" in q or "programmed in" in q or "what language" in q or "which language" in q:
        return "language"
    return "generic"

def exact_answer_from_sentence(question: str, sentence: str) -> str | None:
    q = question.lower()
    s = sentence.strip()

    if question_type(question) == "year":
        y = extract_year(s)
        if y:
            return f"{re.sub(r'\\s+', ' ', s.split(' in ')[0]).strip()} in {y}." if " in " in s else s
        return None

    if "what year was" in q or "when was" in q:
        y = extract_year(s)
        if y:
            return f"{re.sub(r'\\s+', ' ', s).strip()}"

    if any(x in q for x in ["written in", "what language", "which language"]):
        if any(x in s.lower() for x in ["written in", "implemented in", "programmed in", "rust", "python", "java", "javascript", "go", "c++"]):
            return s

    if any(x in q for x in ["released", "open-sourced", "launched", "developed", "created", "built", "founded"]):
        return s

    return None

def strong_support(question: str, sentence: str) -> bool:
    q_tokens = tokenize(question)
    s_tokens = tokenize(sentence)
    overlap = q_tokens & s_tokens
    return len(overlap) >= 2 and (extract_year(sentence) is not None or len(sentence) < 120)

@app.post("/grounded-qa")
async def grounded_qa(payload: QARequest):
    try:
        question = normalize(payload.question or "")
        chunks = payload.chunks or []

        if not question or not chunks:
            return unanswerable_response()

        valid_chunks = [
            c for c in chunks
            if isinstance(c.chunk_id, str) and c.chunk_id.strip()
            and isinstance(c.text, str) and c.text.strip()
        ]
        if not valid_chunks:
            return unanswerable_response()

        best_chunk, best_sentence, best_score = find_best_support(question, valid_chunks)

        # Be stricter than before
        if best_chunk is None or best_sentence is None or best_score < 0.50:
            return unanswerable_response()

        # Only use the exact sentence from the cited chunk
        answer = best_sentence.strip()

        # Hard support check: answer must exactly be a sentence in that same chunk
        sentences_in_chunk = [s.strip() for s in split_sentences(best_chunk.text)]
        if answer not in sentences_in_chunk:
            return unanswerable_response()

        return {
            "answer": answer,
            "citations": [best_chunk.chunk_id],
            "confidence": 0.9,
            "answerable": True,
        }

    except Exception:
        return unanswerable_response()

@app.get("/")
async def health_check():
    return {"status": "ok"}