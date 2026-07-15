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


def extract_subject(question: str) -> str:
    q = normalize(question).rstrip("?.")

    patterns = [
        r"what year was (.+?) released$",
        r"which year was (.+?) released$",
        r"when was (.+?) released$",
        r"what year was (.+?) open[- ]sourced$",
        r"when was (.+?) open[- ]sourced$",
        r"who developed (.+)$",
        r"who created (.+)$",
        r"who built (.+)$",
        r"what language is (.+?) written in$",
        r"which language is (.+?) written in$",
        r"what is (.+)$",
    ]

    for p in patterns:
        m = re.match(p, q, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return ""


def subject_in_sentence(subject: str, sentence: str) -> bool:
    if not subject:
        return False

    subject_tokens = [t for t in re.findall(r"[a-z0-9]+", subject.lower()) if len(t) > 2]
    sentence_lower = sentence.lower()

    if not subject_tokens:
        return False

    return all(token in sentence_lower for token in subject_tokens)


def classify_question(question: str) -> str:
    q = question.lower()

    if "what year" in q or "which year" in q or q.startswith("when "):
        return "year"
    if q.startswith("who "):
        return "who"
    if "written in" in q or "what language" in q or "which language" in q:
        return "language"
    if q.startswith("what is "):
        return "definition"

    return "unsupported"


def sentence_answers_question(question: str, sentence: str) -> bool:
    qtype = classify_question(question)
    s = sentence.lower()

    if qtype == "year":
        return (
            extract_year(sentence) is not None and
            any(x in s for x in ["released", "open-sourced", "open sourced", "launched"])
        )

    if qtype == "who":
        return any(x in s for x in ["developed by", "created by", "built by", "made by"])

    if qtype == "language":
        return any(x in s for x in ["written in", "implemented in", "programmed in"])

    if qtype == "definition":
        return True

    return False


def score_sentence(question: str, sentence: str) -> float:
    q_words = set(re.findall(r"[a-z0-9]+", question.lower()))
    s_words = set(re.findall(r"[a-z0-9]+", sentence.lower()))

    if not q_words or not s_words:
        return 0.0

    overlap = q_words & s_words
    return len(overlap) / len(q_words)


@app.post("/grounded-qa")
async def grounded_qa(payload: QARequest):
    try:
        question = normalize(payload.question or "")
        chunks = payload.chunks or []

        if not question or not chunks:
            return unanswerable_response()

        qtype = classify_question(question)
        if qtype == "unsupported":
            return unanswerable_response()

        subject = extract_subject(question)
        if not subject:
            return unanswerable_response()

        best_chunk_id = None
        best_sentence = None
        best_score = 0.0

        for chunk in chunks:
            if not chunk.chunk_id or not chunk.text:
                continue

            for sentence in split_sentences(chunk.text):
                if not subject_in_sentence(subject, sentence):
                    continue
                if not sentence_answers_question(question, sentence):
                    continue

                score = score_sentence(question, sentence)
                if score > best_score:
                    best_score = score
                    best_chunk_id = chunk.chunk_id
                    best_sentence = sentence

        if not best_chunk_id or not best_sentence or best_score < 0.25:
            return unanswerable_response()

        confidence = round(min(0.95, 0.55 + best_score * 0.3), 2)

        return {
            "answer": best_sentence,
            "citations": [best_chunk_id],
            "confidence": confidence,
            "answerable": True
        }

    except Exception:
        return unanswerable_response()


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Grounded QA API is running"}