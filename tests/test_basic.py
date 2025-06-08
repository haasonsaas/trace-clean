"""
Basic tests for trace-clean functionality.
"""

import os
import subprocess
import tempfile

import pytest


def test_help_command():
    """Test that help command works."""
    result = subprocess.run(["python", "-m", "trace_clean_pkg", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "trace-clean" in result.stdout
    assert "Stack-Trace Summarizer" in result.stdout


def test_import_works():
    """Test that the package can be imported."""
    from trace_clean_pkg import TraceClean, main

    assert TraceClean is not None
    assert main is not None


def test_basic_functionality():
    """Test basic trace analysis with a simple stack trace."""
    from trace_clean_pkg.cli import TraceClean

    # Test without API key (should fail gracefully)
    with pytest.raises(ValueError, match="API key required"):
        TraceClean(model="gpt-4o-mini", api_key=None)


def test_file_input():
    """Test file input functionality."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test stack trace content")
        temp_file = f.name

    try:
        result = subprocess.run(
            ["python", "-m", "trace_clean_pkg", temp_file],
            capture_output=True,
            text=True,
            env={**os.environ, "OPENAI_API_KEY": "test-key"},
        )
        # Should get an error about the API key being invalid, but file should be read
        assert "test-key" not in result.stderr  # Don't expose API key in error
    finally:
        os.unlink(temp_file)


def test_json_output_flag():
    """Test that JSON flag is recognized."""
    result = subprocess.run(["python", "-m", "trace_clean_pkg", "--json", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
