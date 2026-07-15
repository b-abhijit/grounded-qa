import re
from typing import List, Optional, Set, Tuple

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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def tokenize(text: str) -> Set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", normalize_text(text))
    return [p for p in parts if p]


def unanswerable_response():
    return {
        "answer": "I don't know",
        "citations": [],
        "confidence": 0.1,
        "answerable": False
    }


def extract_year(text: str) -> Optional[str]:
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return m.group(0) if m else None


def question_type(question: str) -> str:
    q = question.lower().strip()
    if "what year" in q or "which year" in q or q.startswith("when "):
        return "year"
    if q.startswith("who ") or " who " in q:
        return "who"
    if q.startswith("where ") or " where " in q:
        return "where"
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


def extract_subject_tokens(question: str) -> Set[str]:
    q = question.lower().strip(" ?.")
    patterns = [
        r"what year was (.+)",
        r"which year was (.+)",
        r"when was (.+)",
        r"who developed (.+)",
        r"who created (.+)",
        r"who built (.+)",
        r"who founded (.+)",
        r"what is (.+)",
        r"where is (.+)",
        r"where was (.+)",
        r"what language is (.+) written in",
        r"which language is (.+) written in"
    ]

    for pattern in patterns:
        m = re.match(pattern, q)
        if m:
            return tokenize(m.group(1))

    return tokenize(q)


def subject_supported(question: str, sentence: str) -> bool:
    subject_tokens = extract_subject_tokens(question)
    sentence_tokens = tokenize(sentence)

    if not subject_tokens:
        return False

    overlap = subject_tokens & sentence_tokens
    return len(overlap) >= max(1, min(2, len(subject_tokens)))


def relation_supported(rel_type: str, sentence: str) -> bool:
    s = sentence.lower()

    relation_map = {
        "release": ["released", "open-sourced", "open sourced", "launched"],
        "developed": ["developed by", "created by", "built by", "made by", "developed"],
        "language": ["written in", "implemented in", "programmed in"],
        "founded": ["founded by", "founder", "founded"],
        "generic": []
    }

    if rel_type == "generic":
        return True

    return any(cue in s for cue in relation_map.get(rel_type, []))


def answer_type_supported(q_type: str, sentence: str) -> bool:
    s = sentence.strip()

    if q_type == "year":
        return extract_year(s) is not None

    if q_type == "who":
        return bool(re.search(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", s)) or \
               any(x in s.lower() for x in ["research", "university", "inc", "corp", "lab", "laboratory"])

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

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


def candidate_supported(question: str, sentence: str) -> bool:
    q_type = question_type(question)
    rel_type = relation_type(question)

    return (
        subject_supported(question, sentence) and
        relation_supported(rel_type, sentence) and
        answer_type_supported(q_type, sentence)
    )


def find_best_support(question: str, chunks: List[Chunk]) -> Tuple[Optional[Chunk], Optional[str], float]:
    best_chunk = None
    best_sentence = None
    best_score = 0.0

    for chunk in chunks:
        if not chunk.chunk_id or not chunk.text:
            continue

        for sentence in split_sentences(chunk.text):
            if not candidate_supported(question, sentence):
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
        question = normalize_text(payload.question)
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

        THRESHOLD = 0.50
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
    return {
        "status": "ok",
        "message": "Grounded QA API is running"
    }