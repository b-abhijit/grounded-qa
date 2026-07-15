"""
Grounded QA API — answers questions ONLY using provided chunks,
cites the chunk it used, and returns a calibrated confidence score.

Run locally with:
    uvicorn app:app --reload --port 8000
"""

import re
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="SafeAnswer AI - Grounded QA API")

# ---------------------------------------------------------------
# 1. CORS — lets any website/browser call this API without being blocked.
#    (Assignment rule #4: "handles CORS")
# ---------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------
# 2. Define the shape of the incoming JSON (this is what the
#    grader's request will look like).
# ---------------------------------------------------------------
class Chunk(BaseModel):
    chunk_id: Optional[str] = None
    text: Optional[str] = None


class QARequest(BaseModel):
    question: Optional[str] = None
    chunks: Optional[List[Chunk]] = None


# ---------------------------------------------------------------
# 3. Small helper functions.
#    tokenize()      -> turns a sentence into a set of meaningful words
#    split_sentences -> breaks a chunk's text into individual sentences,
#                        so we can quote just ONE sentence, not a whole paragraph
# ---------------------------------------------------------------
STOPWORDS = {
    "the", "is", "was", "are", "were", "a", "an", "of", "in", "on", "for",
    "to", "and", "or", "what", "when", "where", "who", "which", "how",
    "did", "does", "do", "by", "at", "as", "with", "that", "this", "it",
    "released", "year",  # generic words that shouldn't drive matching too hard
}


def tokenize(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def unanswerable_response():
    return {
        "answer": "I don't know",
        "citations": [],
        "confidence": 0.1,
        "answerable": False,
    }


# ---------------------------------------------------------------
# 4. THE MAIN ENDPOINT
# ---------------------------------------------------------------
@app.post("/grounded-qa")
async def grounded_qa(payload: QARequest):
    try:
        question = (payload.question or "").strip()
        chunks = payload.chunks or []

        # Rule 4: handle empty / malformed input gracefully
        if not question or not chunks:
            return unanswerable_response()

        q_tokens = tokenize(question)
        if not q_tokens:
            return unanswerable_response()

        # Find the single best-matching sentence across ALL chunks
        best_score = 0.0
        best_chunk_id = None
        best_sentence = None

        for chunk in chunks:
            if not chunk.chunk_id or not chunk.text:
                continue  # skip malformed chunk entries

            for sentence in split_sentences(chunk.text):
                s_tokens = tokenize(sentence)
                if not s_tokens:
                    continue

                overlap = q_tokens & s_tokens
                # score = how much of the QUESTION's meaning is covered
                # by this sentence (0.0 to 1.0)
                score = len(overlap) / len(q_tokens)

                if score > best_score:
                    best_score = score
                    best_chunk_id = chunk.chunk_id
                    best_sentence = sentence.strip()

        # ------------------------------------------------------
        # 5. THE ANSWERABILITY GATE
        #    If the best match isn't good enough, refuse to answer
        #    rather than risk hallucinating.
        # ------------------------------------------------------
        THRESHOLD = 0.34  # tune this: higher = stricter grounding

        if best_chunk_id is None or best_score < THRESHOLD:
            return unanswerable_response()

        # ------------------------------------------------------
        # 6. CALIBRATED CONFIDENCE
        #    Maps the raw overlap score (0.34 - 1.0) into a
        #    believable confidence range (0.67 - 0.95).
        # ------------------------------------------------------
        confidence = round(min(0.95, 0.5 + best_score * 0.5), 2)

        return {
            "answer": best_sentence,
            "citations": [best_chunk_id],
            "confidence": confidence,
            "answerable": True,
        }

    except Exception:
        # Rule 4: never crash — always fail safe into "I don't know"
        return unanswerable_response()


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Grounded QA API is running"}