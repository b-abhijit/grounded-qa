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


def tokenize(text: str) -> Set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def unanswerable_response():
    return {
        "answer": "I don't know",
        "citations": [],
        "confidence": 0.1,
        "answerable": False
    }


def extract_year(text: str) -> Optional[str]:
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else None


def question_type(question: str) -> str:
    q = question.lower().strip()
    if "what year" in q or "which year" in q or q.startswith("when "):
        return "year"
    if q.startswith("who ") or " who " in q:
        return "who"
    if q.startswith("where ") or " where " in q:
        return "where"
    return "generic"


def explicit_support(q_type: str, sentence: str) -> bool:
    s = sentence.strip()

    if q_type == "year":
        return extract_year(s) is not None

    if q_type == "who":
        return bool(re.search(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", s)) or \
               any(x in s.lower() for x in ["research", "university", "inc", "corp", "laboratory", "lab"])

    if q_type == "where":
        return any(x in s.lower() for x in [" in ", " at ", " from ", " based in ", " located in "])

    return True


def sentence_score(question: str, sentence: str) -> float:
    q_tokens = tokenize(question)
    s_tokens = tokenize(sentence)

    if not q_tokens or not s_tokens:
        return 0.0

    overlap = q_tokens & s_tokens
    precision = len(overlap) / len(s_tokens)
    recall = len(overlap) / len(q_tokens)

    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0


def find_best_support(question: str, chunks: List[Chunk]):
    q_type = question_type(question)
    best_chunk = None
    best_sentence = None
    best_score = 0.0

    for chunk in chunks:
        if not chunk.chunk_id or not chunk.text:
            continue

        for sentence in split_sentences(chunk.text):
            sent = normalize_text(sentence)
            score = sentence_score(question, sent)

            if score > best_score and explicit_support(q_type, sent):
                best_score = score
                best_chunk = chunk
                best_sentence = sent

    return best_chunk, best_sentence, best_score


@app.post("/grounded-qa")
async def grounded_qa(payload: QARequest):
    try:
        question = normalize_text(payload.question or "")
        chunks = payload.chunks or []

        if not question or not chunks:
            return unanswerable_response()

        valid_chunks = [
            c for c in chunks
            if c.chunk_id and isinstance(c.chunk_id, str) and c.text and isinstance(c.text, str)
        ]
        if not valid_chunks:
            return unanswerable_response()

        best_chunk, best_sentence, best_score = find_best_support(question, valid_chunks)

        THRESHOLD = 0.55
        if best_chunk is None or best_sentence is None or best_score < THRESHOLD:
            return unanswerable_response()

        confidence = round(min(0.95, 0.35 + best_score * 0.5), 2)

        return {
            "answer": best_sentence,
            "citations": [best_chunk.chunk_id],
            "confidence": confidence,
            "answerable": True
        }

    except Exception:
        return unanswerable_response()


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Grounded QA API is running"}