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
    "the", "is", "are", "were", "was", "be", "been", "being", "a", "an", "of",
    "in", "on", "for", "to", "and", "or", "what", "when", "where", "who",
    "whom", "which", "how", "many", "much", "did", "does", "do", "by", "at",
    "as", "with", "that", "this", "it", "from", "into", "their", "there",
    "have", "has", "had", "you", "your", "please", "tell", "me", "can"
}

# words too generic to ever count as a real signal on their own
GENERIC_WORDS: Set[str] = {"year", "name", "type", "kind", "amount", "number"}


def stem(word: str) -> str:
    """Very small suffix stripper so 'released'/'release', 'developed'/'develops' etc match."""
    for suf in ("ational", "ization", "ing", "edly", "ed", "es", "ies"):
        if len(word) > len(suf) + 3 and word.endswith(suf):
            return word[: -len(suf)]
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> Set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {stem(w) for w in words if w not in STOPWORDS}


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


def extract_number(text: str) -> Optional[str]:
    match = re.search(r"\b\d+(\.\d+)?\b", text)
    return match.group(0) if match else None


def has_proper_noun(text: str) -> bool:
    # crude check for a capitalized word that isn't sentence-initial-only
    words = text.split()
    return any(w[:1].isupper() for w in words[1:]) or (words and words[0][:1].isupper())


def sentence_score(question: str, sentence: str) -> float:
    q_tokens = tokenize(question)
    s_tokens = tokenize(sentence)

    if not q_tokens or not s_tokens:
        return 0.0

    overlap = q_tokens & s_tokens
    # ignore near-empty overlaps on very short questions to avoid one-word flukes
    meaningful_overlap = {t for t in overlap if t not in GENERIC_WORDS}

    base_score = len(overlap) / len(q_tokens)

    # require at least one non-generic overlapping token, unless it's the only token
    if not meaningful_overlap and len(q_tokens) > 1:
        base_score *= 0.5

    q_lower = question.lower()
    if ("what year" in q_lower or "which year" in q_lower or q_lower.startswith("when ")) and extract_year(sentence):
        base_score += 0.2
    if ("how many" in q_lower or "how much" in q_lower) and extract_number(sentence):
        base_score += 0.15
    if q_lower.startswith("who ") and has_proper_noun(sentence):
        base_score += 0.1

    return min(base_score, 1.0)


def find_best_support(question: str, chunks: List[Chunk]):
    best_chunk = None
    best_sentence = None
    best_score = 0.0

    for chunk in chunks:
        if not chunk.chunk_id or not chunk.text:
            continue

        for sentence in split_sentences(chunk.text):
            score = sentence_score(question, sentence)
            if score > best_score:
                best_score = score
                best_chunk = chunk
                best_sentence = normalize_text(sentence)

    return best_chunk, best_sentence, best_score


def answer_is_supported(question: str, answer: str) -> bool:
    q_lower = question.lower()

    if "what year" in q_lower or "which year" in q_lower or q_lower.startswith("when "):
        return extract_year(answer) is not None

    if "how many" in q_lower or "how much" in q_lower:
        return extract_number(answer) is not None

    if q_lower.startswith("who "):
        return has_proper_noun(answer)

    return True


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

        THRESHOLD = 0.6
        if best_chunk is None or best_sentence is None or best_score < THRESHOLD:
            return unanswerable_response()

        if not answer_is_supported(question, best_sentence):
            return unanswerable_response()

        confidence = round(min(0.95, 0.45 + best_score * 0.45), 2)

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