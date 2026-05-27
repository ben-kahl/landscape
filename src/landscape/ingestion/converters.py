"""Convert source files to the markdown text the ingestion pipeline expects.

Plain text and markdown bypass markitdown entirely (identity conversion would
just add startup latency). Everything else goes through markitdown's Python
API. Unknown extensions fall back to a utf-8 read with source_type="text" so
the caller can still feed in arbitrary text files without special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Maps extension (lowercase, with leading dot) to the source_type string we
# record on Document nodes. The source_type is the *input* format, not the
# converted format — it tells downstream queries that a fact came from a PDF
# even though the chunker only saw markdown.
_EXTENSION_TO_SOURCE_TYPE: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".xls": "xls",
    ".html": "html",
    ".htm": "html",
    ".csv": "csv",
    ".json": "json",
    ".xml": "xml",
    ".epub": "epub",
    ".rtf": "rtf",
}

# Extensions we read directly as utf-8 without invoking markitdown. Markitdown
# can technically convert these (it's identity for markdown) but spinning up
# the converter is wasted work.
_PASSTHROUGH_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown", ".txt"})


class ConverterError(RuntimeError):
    """Raised when a file cannot be read or converted to markdown."""


@dataclass(frozen=True)
class ConvertedDocument:
    """Result of converting a source file to markdown."""

    text: str
    source_type: str
    title_hint: str | None


def supported_extensions() -> frozenset[str]:
    """Extensions for which we have a known converter or passthrough."""
    return frozenset(_EXTENSION_TO_SOURCE_TYPE.keys())


def is_supported_extension(path: Path) -> bool:
    return path.suffix.lower() in _EXTENSION_TO_SOURCE_TYPE


def convert_to_markdown(path: Path) -> ConvertedDocument:
    """Read ``path`` and return its markdown representation plus provenance.

    Dispatch:
      - ``.md`` / ``.markdown`` / ``.txt`` → utf-8 read, no conversion.
      - Other known extensions → markitdown's Python API.
      - Unknown extensions → utf-8 read with source_type="text". Lets callers
        ingest arbitrary plain-text files (logs, source code) without us
        having to maintain an exhaustive allowlist.

    Raises ``ConverterError`` if the file cannot be read or markitdown fails
    on a recognized format. Callers (CLI / pipeline) surface this as a clean
    error rather than letting markitdown's internal exceptions escape.
    """
    if not path.exists():
        raise ConverterError(f"path does not exist: {path}")
    if not path.is_file():
        raise ConverterError(f"path is not a file: {path}")

    ext = path.suffix.lower()
    source_type = _EXTENSION_TO_SOURCE_TYPE.get(ext, "text")

    if ext in _PASSTHROUGH_EXTENSIONS or ext not in _EXTENSION_TO_SOURCE_TYPE:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ConverterError(f"cannot read {path} as utf-8: {exc}") from exc
        return ConvertedDocument(text=text, source_type=source_type, title_hint=None)

    # Lazy import: markitdown pulls in pdfminer/mammoth/etc. and we don't want
    # to load them at module import time for callers that only ingest text.
    try:
        from markitdown import MarkItDown
    except ImportError as exc:  # pragma: no cover — dependency is required
        raise ConverterError(
            "markitdown is required for non-text file ingestion"
        ) from exc

    try:
        result = MarkItDown().convert(str(path))
    except Exception as exc:
        raise ConverterError(f"markitdown failed to convert {path}: {exc}") from exc

    markdown_text = getattr(result, "markdown", None) or getattr(
        result, "text_content", ""
    )
    if not isinstance(markdown_text, str) or not markdown_text.strip():
        raise ConverterError(f"markitdown produced empty output for {path}")

    title_hint = getattr(result, "title", None)
    if isinstance(title_hint, str):
        title_hint = title_hint.strip() or None
    else:
        title_hint = None

    return ConvertedDocument(
        text=markdown_text,
        source_type=source_type,
        title_hint=title_hint,
    )
