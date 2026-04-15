import re
from dataclasses import dataclass

# Split on sentence-ending punctuation followed by whitespace + capital letter
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
TARGET_TOKENS = 300
MAX_TOKENS = 400


@dataclass
class Chunk:
    index: int
    text: str


def chunk_text(text: str) -> list[Chunk]:
    sentences = SENTENCE_SPLIT.split(text.strip())
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for sent in sentences:
        sent_tokens = len(sent.split())
        # Flush buffer when adding this sentence would exceed MAX_TOKENS
        if buf_tokens + sent_tokens > MAX_TOKENS and buf:
            chunks.append(" ".join(buf))
            buf = []
            buf_tokens = 0
        buf.append(sent)
        buf_tokens += sent_tokens
    if buf:
        chunks.append(" ".join(buf))
    return [Chunk(index=i, text=t) for i, t in enumerate(chunks)]
