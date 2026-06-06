"""Guards that importing the MCP app does not eagerly import the pipeline.

mcp_app.py deliberately uses function-local imports for heavy modules so that
loading the MCP server (e.g. for tool registration) stays cheap. Preserved from
the now-removed test_capture_cutover.py, which otherwise only covered the
deleted push-capture buffer.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit


def test_mcp_app_import_does_not_load_pipeline():
    code = textwrap.dedent(
        """
        import sys
        import landscape.mcp_app
        raise SystemExit(1 if "landscape.pipeline" in sys.modules else 0)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
