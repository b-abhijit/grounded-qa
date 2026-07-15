import re
from typing import List, Optional, Set

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


STOPWORDS: Set[str] = {
    "the", "is", "are", "were", "a", "an", "of", "in", "on", "for",
    "to", "and", "or", "what", "when", "where", "who", "which", "how",
    "did", "does", "do", "by", "at", "as", "with", "that", "this", "it",
    "from", "into", "their", "there", "been", "being", "have", "has", "had"
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalize(text)) if s.strip()]


def tokenize(text: str) -> Set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def extract_year(text: str) -> Optional[str]:
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return m.group(0) if m else None


def unanswerable_response():
    return {
        "answer": "I don't know",
        "citations": [],
        "confidence": 0.1,
        "answerable": False,
    }


def question_type(question: str) -> str:
    q = question.lower()
    if "what year" in q or "which year" in q or q.startswith("when "):
        return "year"
    if q.startswith("who ") or " who " in q:
        return "who"
    if "written in" in q or "programmed in" in q or "what language" in q or "which language" in q:
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


def answer_type_supported(q_type: str, sentence: str) -> bool:
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
    overlap = q_tokens & s_tokens
    required = [t for t in q_tokens if len(t) > 2]
    if not required:
        return False
    return len(overlap) >= max(1, min(2, len(required)))


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
    if extract_year(sentence) and question_type(question) == "year":
        score += 0.15
    return min(score, 1.0)


def exact_support_check(answer: str, chunk_text: str) -> bool:
    return normalize(answer).lower() in normalize(chunk_text).lower()


def find_best_support(question: str, chunks: List[Chunk]):
    best_chunk = None
    best_sentence = None
    best_score = 0.0
    q_type = question_type(question)
    rel_type = relation_type(question)
    for chunk in chunks:
        if not chunk.chunk_id or not chunk.text:
            continue
        for sentence in split_sentences(chunk.text):
            if not subject_overlap(question, sentence):
                continue
            if not relation_supported(rel_type, sentence):
                continue
            if not answer_type_supported(q_type, sentence):
                continue
            score = sentence_score(question, sentence)
            if score > best_score:
                best_score = score
                best_chunk = chunk
                best_sentence = sentence
    return best_chunk, best_sentence, best_score


@app.post("/grounded-qa")
async def grounded_qa(payload: QARequest):
    try:
        question = normalize(payload.question or "")
        chunks = payload.chunks or []
        if not question or not chunks:
            return unanswerable_response()
        valid_chunks = [c for c in chunks if isinstance(c.chunk_id, str) and c.chunk_id.strip() and isinstance(c.text, str) and c.text.strip()]
        if not valid_chunks:
            return unanswerable_response()
        best_chunk, best_sentence, best_score = find_best_support(question, valid_chunks)
        if best_chunk is None or best_sentence is None or best_score < 0.34:
            return unanswerable_response()
        answer = best_sentence.strip()
        if not exact_support_check(answer, best_chunk.text):
            return unanswerable_response()
        return {
            "answer": answer,
            "citations": [best_chunk.chunk_id],
            "confidence": round(min(0.95, 0.45 + best_score * 0.45), 2),
            "answerable": True,
        }
    except Exception:
        return unanswerable_response()


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Grounded QA API is running"}