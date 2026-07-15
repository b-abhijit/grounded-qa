import re
from typing import List, Optional

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


def unanswerable_response():
    return {
        "answer": "I don't know",
        "citations": [],
        "confidence": 0.1,
        "answerable": False
    }


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalize(text)) if s.strip()]


def extract_year(text: str) -> Optional[str]:
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return m.group(0) if m else None


def question_subject(question: str) -> str:
    q = normalize(question).rstrip("?.")
    patterns = [
        r"what year was (.+?) released$",
        r"what year was (.+?) open[- ]sourced$",
        r"when was (.+?) released$",
        r"who developed (.+)$",
        r"who created (.+)$",
        r"who built (.+)$",
        r"what is (.+)$",
        r"which language is (.+?) written in$",
        r"what language is (.+?) written in$",
    ]
    for p in patterns:
        m = re.match(p, q, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def subject_in_sentence(subject: str, sentence: str) -> bool:
    if not subject:
        return False
    subject_tokens = re.findall(r"[a-z0-9]+", subject.lower())
    sent_lower = sentence.lower()
    important = [t for t in subject_tokens if len(t) > 2]
    if not important:
        return False
    return all(t in sent_lower for t in important)


def answer_release_year(question: str, chunks: List[Chunk]):
    subject = question_subject(question)
    for chunk in chunks:
        if not chunk.chunk_id or not chunk.text:
            continue
        for sentence in split_sentences(chunk.text):
            s = sentence.lower()
            if subject_in_sentence(subject, sentence) and (
                "released" in s or "open-sourced" in s or "open sourced" in s or "launched" in s
            ):
                year = extract_year(sentence)
                if year:
                    return {
                        "answer": sentence,
                        "citations": [chunk.chunk_id],
                        "confidence": 0.95,
                        "answerable": True
                    }
    return None


def answer_developed_by(question: str, chunks: List[Chunk]):
    subject = question_subject(question)
    for chunk in chunks:
        if not chunk.chunk_id or not chunk.text:
            continue
        for sentence in split_sentences(chunk.text):
            s = sentence.lower()
            if subject_in_sentence(subject, sentence) and (
                "developed by" in s or "created by" in s or "built by" in s or "made by" in s
            ):
                return {
                    "answer": sentence,
                    "citations": [chunk.chunk_id],
                    "confidence": 0.9,
                    "answerable": True
                }
    return None


def answer_written_in(question: str, chunks: List[Chunk]):
    subject = question_subject(question)
    for chunk in chunks:
        if not chunk.chunk_id or not chunk.text:
            continue
        for sentence in split_sentences(chunk.text):
            s = sentence.lower()
            if subject_in_sentence(subject, sentence) and (
                "written in" in s or "implemented in" in s or "programmed in" in s
            ):
                return {
                    "answer": sentence,
                    "citations": [chunk.chunk_id],
                    "confidence": 0.9,
                    "answerable": True
                }
    return None


@app.post("/grounded-qa")
async def grounded_qa(payload: QARequest):
    try:
        question = normalize(payload.question or "")
        chunks = payload.chunks or []

        if not question or not chunks:
            return unanswerable_response()

        valid_chunks = [
            c for c in chunks
            if isinstance(c.chunk_id, str) and c.chunk_id.strip() and isinstance(c.text, str) and c.text.strip()
        ]
        if not valid_chunks:
            return unanswerable_response()

        q = question.lower()

        if "what year" in q or "when" in q:
            result = answer_release_year(question, valid_chunks)
            return result if result else unanswerable_response()

        if "who developed" in q or "who created" in q or "who built" in q:
            result = answer_developed_by(question, valid_chunks)
            return result if result else unanswerable_response()

        if "written in" in q or "programmed in" in q or "what language" in q or "which language" in q:
            result = answer_written_in(question, valid_chunks)
            return result if result else unanswerable_response()

        return unanswerable_response()

    except Exception:
        return unanswerable_response()


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Grounded QA API is running"}