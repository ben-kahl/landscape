from dataclasses import dataclass

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from transformers import AutoTokenizer

from landscape.config import settings

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

_tokenizer = None
_recursive_splitter: RecursiveCharacterTextSplitter | None = None
_md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=_HEADERS_TO_SPLIT_ON,
    return_each_line=False,
    strip_headers=False,
)


@dataclass
class Chunk:
    index: int
    text: str


def _get_splitter() -> RecursiveCharacterTextSplitter:
    global _tokenizer, _recursive_splitter
    if _recursive_splitter is not None:
        return _recursive_splitter
    tok_kwargs: dict = {}
    if settings.hf_token:
        tok_kwargs["token"] = settings.hf_token
    _tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model, **tok_kwargs)

    def count_tokens(text: str) -> int:
        return len(_tokenizer.encode(text, add_special_tokens=False))

    _recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return _recursive_splitter


def chunk_text(text: str) -> list[Chunk]:
    stripped = text.strip()
    if not stripped:
        return []

    header_docs = _md_splitter.split_text(stripped)
    if not header_docs:
        pieces = [stripped]
    else:
        pieces = [doc.page_content for doc in header_docs]

    splitter = _get_splitter()
    final_texts: list[str] = []
    for piece in pieces:
        final_texts.extend(splitter.split_text(piece))

    return [Chunk(index=i, text=t) for i, t in enumerate(final_texts) if t.strip()]
