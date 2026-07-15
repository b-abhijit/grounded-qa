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
    "the", "is", "was", "are", "were", "a", "an", "of", "in", "on", "for",
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


def best_chunk_and_sentence(question: str, chunks: List[Chunk]):
    q_tokens = tokenize(question)
    if not q_tokens:
        return None, None, 0.0

    best_chunk = None
    best_sentence = None
    best_score = 0.0

    for chunk in chunks:
        if not chunk.chunk_id or not chunk.text:
            continue

        for sentence in split_sentences(chunk.text):
            s_tokens = tokenize(sentence)
            if not s_tokens:
                continue

            overlap = q_tokens & s_tokens
            score = len(overlap) / max(len(q_tokens), 1)

            if score > best_score:
                best_score = score
                best_chunk = chunk
                best_sentence = sentence.strip()

    return best_chunk, best_sentence, best_score


def build_grounded_answer(question: str, sentence: str) -> Optional[str]:
    q_lower = question.lower().strip()

    if "what year" in q_lower or "which year" in q_lower or q_lower.startswith("when "):
        year = extract_year(sentence)
        if not year:
            return None

        subject_match = re.search(
            r"(?:what year was|which year was|when was)\s+(.+?)(?:\?|$)",
            q_lower
        )
        if subject_match:
            subject = subject_match.group(1).strip()
            if subject:
                return f"{subject.upper() if subject.isupper() else subject} was released in {year}."
        return year

    return sentence


@app.post("/grounded-qa")
async def grounded_qa(payload: QARequest):
    try:
        question = (payload.question or "").strip()
        chunks = payload.chunks or []

        if not question or not chunks:
            return unanswerable_response()

        valid_chunks = [
            c for c in chunks
            if c.chunk_id and isinstance(c.chunk_id, str) and c.text and isinstance(c.text, str)
        ]
        if not valid_chunks:
            return unanswerable_response()

        best_chunk, best_sentence, best_score = best_chunk_and_sentence(question, valid_chunks)

        THRESHOLD = 0.5
        if best_chunk is None or best_sentence is None or best_score < THRESHOLD:
            return unanswerable_response()

        answer = build_grounded_answer(question, best_sentence)
        if not answer:
            return unanswerable_response()

        if answer != best_sentence:
            answer_tokens = tokenize(answer)
            sentence_tokens = tokenize(best_sentence)
            if not answer_tokens.issubset(sentence_tokens) and not extract_year(best_sentence):
                return unanswerable_response()

        confidence = round(min(0.95, 0.55 + best_score * 0.35), 2)

        return {
            "answer": answer,
            "citations": [best_chunk.chunk_id],
            "confidence": confidence,
            "answerable": True
        }

    except Exception:
        return unanswerable_response()


@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "message": "Grounded QA API is running"
    }
    