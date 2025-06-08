"""Pytest configuration and shared fixtures for trace-clean tests."""

import os
import sys
from pathlib import Path
from typing import Generator

import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def stacktraces_dir(fixtures_dir: Path) -> Path:
    """Return the path to the stacktraces fixtures directory."""
    return fixtures_dir / "stacktraces"


@pytest.fixture
def sample_python_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a sample Python stacktrace."""
    trace_file = stacktraces_dir / "python" / "sample_traceback.txt"
    return trace_file.read_text()


@pytest.fixture
def sample_javascript_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a sample JavaScript stacktrace."""
    trace_file = stacktraces_dir / "javascript" / "sample_error.txt"
    return trace_file.read_text()


@pytest.fixture
def sample_java_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a sample Java stacktrace."""
    trace_file = stacktraces_dir / "java" / "sample_exception.txt"
    return trace_file.read_text()


@pytest.fixture
def sample_ruby_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a sample Ruby stacktrace."""
    trace_file = stacktraces_dir / "ruby" / "sample_backtrace.txt"
    return trace_file.read_text()


@pytest.fixture
def sample_go_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a sample Go stacktrace."""
    trace_file = stacktraces_dir / "go" / "sample_panic.txt"
    return trace_file.read_text()


@pytest.fixture
def sample_rust_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a sample Rust stacktrace."""
    trace_file = stacktraces_dir / "rust" / "sample_panic.txt"
    return trace_file.read_text()


@pytest.fixture
def sample_cpp_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a sample C++ stacktrace."""
    trace_file = stacktraces_dir / "cpp" / "segfault_trace.txt"
    return trace_file.read_text()


# Additional Python error type fixtures
@pytest.fixture
def python_keyerror_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a Python KeyError stacktrace."""
    trace_file = stacktraces_dir / "python" / "keyerror_trace.txt"
    return trace_file.read_text()


@pytest.fixture
def python_attributeerror_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a Python AttributeError stacktrace."""
    trace_file = stacktraces_dir / "python" / "attributeerror_trace.txt"
    return trace_file.read_text()


@pytest.fixture
def python_indexerror_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a Python IndexError stacktrace."""
    trace_file = stacktraces_dir / "python" / "indexerror_trace.txt"
    return trace_file.read_text()


@pytest.fixture
def python_importerror_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a Python ImportError stacktrace."""
    trace_file = stacktraces_dir / "python" / "importerror_trace.txt"
    return trace_file.read_text()


# Additional Java error type fixtures
@pytest.fixture
def java_classnotfound_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a Java ClassNotFoundException stacktrace."""
    trace_file = stacktraces_dir / "java" / "classnotfound_exception.txt"
    return trace_file.read_text()


@pytest.fixture
def java_concurrent_modification_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a Java ConcurrentModificationException stacktrace."""
    trace_file = stacktraces_dir / "java" / "concurrent_modification_exception.txt"
    return trace_file.read_text()


# Additional JavaScript error type fixtures
@pytest.fixture
def javascript_referenceerror_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a JavaScript ReferenceError stacktrace."""
    trace_file = stacktraces_dir / "javascript" / "reference_error.txt"
    return trace_file.read_text()


@pytest.fixture
def javascript_syntaxerror_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a JavaScript SyntaxError stacktrace."""
    trace_file = stacktraces_dir / "javascript" / "syntax_error.txt"
    return trace_file.read_text()


# Additional Ruby error type fixtures
@pytest.fixture
def ruby_nomethoderror_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a Ruby NoMethodError stacktrace."""
    trace_file = stacktraces_dir / "ruby" / "nomethoderror_trace.txt"
    return trace_file.read_text()


@pytest.fixture
def ruby_argumenterror_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a Ruby ArgumentError stacktrace."""
    trace_file = stacktraces_dir / "ruby" / "argumenterror_trace.txt"
    return trace_file.read_text()


# Additional Go error type fixtures
@pytest.fixture
def go_nil_pointer_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a Go nil pointer panic stacktrace."""
    trace_file = stacktraces_dir / "go" / "nil_pointer_panic.txt"
    return trace_file.read_text()


# Additional Rust error type fixtures
@pytest.fixture
def rust_unwrap_panic_stacktrace(stacktraces_dir: Path) -> str:
    """Load and return a Rust unwrap panic stacktrace."""
    trace_file = stacktraces_dir / "rust" / "unwrap_panic.txt"
    return trace_file.read_text()


@pytest.fixture
def temp_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary file for testing."""
    temp_file = tmp_path / "test_stacktrace.txt"
    yield temp_file
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def mock_clipboard(monkeypatch):
    """Mock clipboard functionality for testing."""
    clipboard_content = None
    
    def mock_copy(text):
        nonlocal clipboard_content
        clipboard_content = text
    
    def mock_paste():
        return clipboard_content or ""
    
    monkeypatch.setattr("pyperclip.copy", mock_copy)
    monkeypatch.setattr("pyperclip.paste", mock_paste)
    
    return {"copy": mock_copy, "paste": mock_paste, "content": lambda: clipboard_content}


@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    from click.testing import CliRunner
    return CliRunner()


@pytest.fixture
def mock_openai_response(monkeypatch):
    """Mock OpenAI API responses for testing."""
    def mock_response(prompt, model="gpt-4", temperature=0):
        # Return a predictable cleaned stacktrace
        return """Here's the cleaned stacktrace:

Error: Division by zero
  at calculate() in calculator.py:42
  at main() in app.py:15"""
    
    # This will need to be adjusted based on how the actual OpenAI integration is implemented
    monkeypatch.setattr("openai.ChatCompletion.create", lambda **kwargs: {
        "choices": [{"message": {"content": mock_response(kwargs.get("messages", [{}])[-1].get("content", ""))}}]
    })
    
    return mock_response