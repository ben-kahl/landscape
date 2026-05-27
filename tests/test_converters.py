"""Unit tests for landscape.ingestion.converters.

These tests build small fixtures on the fly rather than checking binary blobs
into the repo. Each test produces the minimum content needed to verify that
markitdown's conversion preserves a feature we care about downstream (headings,
tables, etc.).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from landscape.ingestion.converters import (
    ConverterError,
    convert_to_markdown,
    is_supported_extension,
    supported_extensions,
)

pytestmark = pytest.mark.unit


def test_plain_text_passthrough(tmp_path: Path) -> None:
    p = tmp_path / "notes.txt"
    p.write_text("hello world\n", encoding="utf-8")

    converted = convert_to_markdown(p)

    assert converted.text == "hello world\n"
    assert converted.source_type == "text"
    assert converted.title_hint is None


def test_markdown_passthrough_keeps_source_type(tmp_path: Path) -> None:
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nbody text\n", encoding="utf-8")

    converted = convert_to_markdown(p)

    assert "# Title" in converted.text
    assert converted.source_type == "markdown"


def test_unknown_extension_falls_back_to_text(tmp_path: Path) -> None:
    p = tmp_path / "server.log"
    p.write_text("2026-05-27 INFO startup ok\n", encoding="utf-8")

    converted = convert_to_markdown(p)

    assert "startup ok" in converted.text
    assert converted.source_type == "text"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConverterError, match="path does not exist"):
        convert_to_markdown(tmp_path / "no-such-file.pdf")


def test_directory_argument_raises(tmp_path: Path) -> None:
    with pytest.raises(ConverterError, match="not a file"):
        convert_to_markdown(tmp_path)


def test_non_utf8_text_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "bytes.txt"
    p.write_bytes(b"\xff\xfe\x00not-utf8")

    with pytest.raises(ConverterError, match="utf-8"):
        convert_to_markdown(p)


def test_html_table_becomes_markdown_table(tmp_path: Path) -> None:
    html = """
    <html><body>
    <h1>Quarterly Numbers</h1>
    <table>
        <tr><th>Quarter</th><th>Revenue</th></tr>
        <tr><td>Q1</td><td>100</td></tr>
        <tr><td>Q2</td><td>140</td></tr>
    </table>
    </body></html>
    """
    p = tmp_path / "report.html"
    p.write_text(html, encoding="utf-8")

    converted = convert_to_markdown(p)

    assert converted.source_type == "html"
    assert "Quarterly Numbers" in converted.text
    # Markitdown emits pipe-style markdown tables. Don't assert exact
    # whitespace — assert the values and the pipe separator are present.
    assert "Quarter" in converted.text
    assert "Revenue" in converted.text
    assert "Q1" in converted.text
    assert "100" in converted.text
    assert "|" in converted.text


def test_csv_becomes_markdown_table(tmp_path: Path) -> None:
    p = tmp_path / "people.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "role"])
        writer.writerow(["Alice", "Engineer"])
        writer.writerow(["Bob", "Manager"])

    converted = convert_to_markdown(p)

    assert converted.source_type == "csv"
    assert "Alice" in converted.text
    assert "Engineer" in converted.text
    assert "Bob" in converted.text


def test_docx_headings_preserved(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")  # python-docx ships with markitdown[docx]
    doc = docx.Document()
    doc.add_heading("Project Aurora", level=1)
    doc.add_paragraph("Aurora is led by Sarah on the Platform Team.")
    doc.add_heading("Status", level=2)
    doc.add_paragraph("Aurora uses PostgreSQL.")
    p = tmp_path / "aurora.docx"
    doc.save(p)

    converted = convert_to_markdown(p)

    assert converted.source_type == "docx"
    assert "Project Aurora" in converted.text
    assert "Sarah" in converted.text
    assert "PostgreSQL" in converted.text
    # At least one heading marker survived. Markitdown may emit "# " or "## "
    # depending on the level; assert any heading prefix is present.
    assert "#" in converted.text


def test_supported_extensions_includes_common_formats() -> None:
    exts = supported_extensions()

    for required in {".md", ".txt", ".pdf", ".docx", ".html", ".csv"}:
        assert required in exts, f"{required} should be in supported extensions"


def test_is_supported_extension_case_insensitive(tmp_path: Path) -> None:
    assert is_supported_extension(tmp_path / "foo.PDF")
    assert is_supported_extension(tmp_path / "foo.DocX")
    assert not is_supported_extension(tmp_path / "foo.bin")
