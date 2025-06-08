"""Test utilities and helper functions."""

import tempfile
from pathlib import Path
from typing import List, Dict, Any


def create_temp_stacktrace_file(content: str, filename: str = "test_trace.txt") -> Path:
    """Create a temporary file with stacktrace content."""
    temp_dir = Path(tempfile.gettempdir())
    temp_file = temp_dir / filename
    temp_file.write_text(content)
    return temp_file


def assert_stacktrace_cleaned(original: str, cleaned: str) -> bool:
    """Assert that a stacktrace has been properly cleaned."""
    # Basic checks that cleaned version should satisfy
    assert len(cleaned) > 0, "Cleaned stacktrace should not be empty"
    assert len(cleaned) <= len(original), "Cleaned stacktrace should not be longer than original"
    
    # Check that the core error information is preserved
    lines_original = original.split('\n')
    lines_cleaned = cleaned.split('\n')
    
    # At minimum, should preserve the main error type/message
    first_error_line = next((line for line in lines_original if 'Error' in line or 'Exception' in line), "")
    if first_error_line:
        assert any(first_error_line.split(':')[0] in line for line in lines_cleaned), \
            "Main error type should be preserved"
    
    return True


def get_sample_stacktraces() -> Dict[str, str]:
    """Get all sample stacktraces for testing."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "stacktraces"
    
    stacktraces = {}
    for lang_dir in fixtures_dir.iterdir():
        if lang_dir.is_dir():
            for trace_file in lang_dir.glob("*.txt"):
                key = f"{lang_dir.name}_{trace_file.stem}"
                stacktraces[key] = trace_file.read_text()
    
    return stacktraces


def mock_api_response(prompt: str) -> str:
    """Generate a mock API response for testing."""
    if "python" in prompt.lower():
        return """Here's the cleaned Python stacktrace:

TypeError: 'NoneType' object is not iterable
  at check_format() in validators.py:56
  at validate_input() in validators.py:78
  at process_data() in main.py:32

AttributeError: 'TypeError' object has no attribute 'message'
  at log_error() in logging_utils.py:23"""
    
    elif "javascript" in prompt.lower():
        return """Here's the cleaned JavaScript stacktrace:

TypeError: Cannot read property 'length' of undefined
  at calculateTotal() in calculator.js:42
  at processItems() in orderService.js:128
  at OrderController.createOrder() in orderController.js:56

ReferenceError: config is not defined
  at Database.connect() in connection.js:15
  at initializeApp() in app.js:34"""
    
    else:
        return """Here's the cleaned stacktrace:

Generic error occurred
  at main function
  at entry point"""


class MockFileSystem:
    """Mock file system for testing file operations."""
    
    def __init__(self):
        self.files: Dict[str, str] = {}
    
    def write_file(self, path: str, content: str):
        """Mock file write operation."""
        self.files[path] = content
    
    def read_file(self, path: str) -> str:
        """Mock file read operation."""
        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        return self.files[path]
    
    def file_exists(self, path: str) -> bool:
        """Check if mock file exists."""
        return path in self.files
    
    def list_files(self) -> List[str]:
        """List all mock files."""
        return list(self.files.keys())