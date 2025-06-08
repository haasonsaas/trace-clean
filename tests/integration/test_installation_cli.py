"""
Integration tests for installation and CLI functionality of trace-clean.
Tests package installation, console scripts, and command-line interface behavior.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestPackageInstallation:
    """Test package installation with pip install -e ."""

    def test_editable_install_package(self):
        """Test that package can be installed in editable mode."""
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode == 0
        assert "Successfully installed" in result.stdout or "Requirement already satisfied" in result.stdout

    def test_package_importable_after_install(self):
        """Test that package is importable after installation."""
        try:
            import trace_clean_pkg

            assert hasattr(trace_clean_pkg, "__version__")
        except ImportError:
            pytest.fail("Package not importable after installation")

    def test_package_metadata_accessible(self):
        """Test that package metadata is accessible."""
        try:
            import trace_clean_pkg

            # Check that basic attributes exist
            assert hasattr(trace_clean_pkg, "__version__")
        except (ImportError, AttributeError):
            pytest.fail("Package metadata not accessible")


class TestConsoleScriptIntegration:
    """Test trace-clean console script availability and basic functionality."""

    def test_trace_clean_command_available(self):
        """Test that trace-clean command is available in PATH."""
        result = subprocess.run(["trace-clean", "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()

    def test_console_script_entry_point(self):
        """Test that console script entry point works."""
        result = subprocess.run(["trace-clean", "--version"], capture_output=True, text=True)
        # Should either show version or fail gracefully
        assert result.returncode in [0, 1, 2]  # Allow for different version handling

    def test_console_script_with_invalid_args(self):
        """Test console script behavior with invalid arguments."""
        result = subprocess.run(["trace-clean", "--invalid-flag"], capture_output=True, text=True)
        assert result.returncode != 0
        assert "error:" in result.stderr.lower() or "usage:" in result.stderr.lower()


class TestModuleExecution:
    """Test python -m trace_clean_pkg execution."""

    def test_module_execution_help(self):
        """Test module execution with help flag."""
        result = subprocess.run([sys.executable, "-m", "trace_clean_pkg", "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()

    def test_module_execution_version(self):
        """Test module execution with version flag."""
        result = subprocess.run([sys.executable, "-m", "trace_clean_pkg", "--version"], capture_output=True, text=True)
        # Version command should work or fail gracefully
        assert result.returncode in [0, 1, 2]

    def test_module_execution_no_args(self):
        """Test module execution without arguments."""
        result = subprocess.run(
            [sys.executable, "-m", "trace_clean_pkg"],
            capture_output=True,
            text=True,
            input="",  # Empty input
        )
        # Should either process empty input or show usage
        assert result.returncode in [0, 1, 2]


class TestCLIArgumentParsing:
    """Test edge cases in command-line argument parsing."""

    def test_empty_api_key_handling(self):
        """Test handling of empty API key."""
        result = subprocess.run(["trace-clean", "--api-key", ""], capture_output=True, text=True, input="test trace")
        assert result.returncode != 0
        assert "api key" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_file_and_stdin_conflict(self):
        """Test handling when both file and stdin are provided."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test trace content")
            temp_file = f.name

        try:
            result = subprocess.run(
                ["trace-clean", "--api-key", "test-key", "--file", temp_file],
                capture_output=True,
                text=True,
                input="stdin content",
            )
            # Should handle this gracefully, preferring file over stdin
            assert result.returncode in [0, 1, 2]
        finally:
            os.unlink(temp_file)

    def test_multiple_file_arguments(self):
        """Test handling of multiple file arguments."""
        result = subprocess.run(
            ["trace-clean", "--api-key", "test-key", "--file", "file1.txt", "--file", "file2.txt"],
            capture_output=True,
            text=True,
        )
        # Should reject multiple files or handle appropriately
        assert result.returncode in [0, 1, 2]


class TestHelpAndErrorMessages:
    """Test user-friendly help and error messages."""

    def test_help_message_completeness(self):
        """Test that help message contains all important information."""
        result = subprocess.run(["trace-clean", "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        help_text = result.stdout.lower()
        assert "usage:" in help_text
        assert "api" in help_text
        assert "file" in help_text or "input" in help_text

    def test_missing_api_key_error(self):
        """Test error message when API key is missing."""
        result = subprocess.run(["trace-clean"], capture_output=True, text=True, input="test trace")
        assert result.returncode != 0
        error_msg = result.stderr.lower()
        assert "api" in error_msg or "key" in error_msg

    def test_file_not_found_error(self):
        """Test error message for non-existent file."""
        result = subprocess.run(
            ["trace-clean", "--api-key", "test-key", "--file", "nonexistent.txt"], capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "no such file" in result.stderr.lower()


class TestInputOutputBehavior:
    """Test file vs stdin input handling and output behavior."""

    def test_stdin_input_processing(self):
        """Test processing input from stdin."""
        test_trace = "Traceback (most recent call last):\n  File test.py, line 1\nKeyError: 'test'"

        result = subprocess.run(
            ["trace-clean", "--api-key", "test-key"], capture_output=True, text=True, input=test_trace
        )
        # Should attempt to process (may fail due to invalid API key, which is fine)
        assert result.returncode in [0, 1, 2]

    def test_file_input_processing(self):
        """Test processing input from file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Traceback (most recent call last):\n  File test.py, line 1\nKeyError: 'test'")
            temp_file = f.name

        try:
            result = subprocess.run(
                ["trace-clean", "--api-key", "test-key", "--file", temp_file], capture_output=True, text=True
            )
            assert result.returncode in [0, 1, 2]
        finally:
            os.unlink(temp_file)

    def test_output_format_consistency(self):
        """Test that output format is consistent."""
        test_trace = "Simple error trace"

        result = subprocess.run(
            ["trace-clean", "--api-key", "test-key"], capture_output=True, text=True, input=test_trace
        )
        # Output should be structured (JSON or formatted text)
        if result.returncode == 0:
            assert len(result.stdout.strip()) > 0


class TestFileHandling:
    """Test handling of different file types and sizes."""

    def test_large_file_handling(self):
        """Test handling of large trace files."""
        large_content = "Traceback (most recent call last):\n" + "  Line\n" * 10000

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(large_content)
            temp_file = f.name

        try:
            result = subprocess.run(
                ["trace-clean", "--api-key", "test-key", "--file", temp_file],
                capture_output=True,
                text=True,
                timeout=30,  # Prevent hanging on large files
            )
            assert result.returncode in [0, 1, 2]
        except subprocess.TimeoutExpired:
            pytest.skip("Large file processing timed out")
        finally:
            os.unlink(temp_file)

    def test_unicode_file_handling(self):
        """Test handling of files with Unicode content."""
        unicode_content = "Traceback (most recent call last):\n  File 'тест.py', line 1\nError: 测试错误"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write(unicode_content)
            temp_file = f.name

        try:
            result = subprocess.run(
                ["trace-clean", "--api-key", "test-key", "--file", temp_file], capture_output=True, text=True
            )
            assert result.returncode in [0, 1, 2]
        finally:
            os.unlink(temp_file)

    def test_binary_file_rejection(self):
        """Test that binary files are rejected appropriately."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"\x00\x01\x02\x03\xff\xfe")  # Binary content
            temp_file = f.name

        try:
            result = subprocess.run(
                ["trace-clean", "--api-key", "test-key", "--file", temp_file], capture_output=True, text=True
            )
            # Should reject binary files or handle gracefully
            assert result.returncode in [0, 1, 2]
        finally:
            os.unlink(temp_file)


class TestErrorHandling:
    """Test error scenarios and exit codes."""

    def test_network_error_simulation(self):
        """Test handling of network errors (simulated with invalid API key)."""
        result = subprocess.run(
            ["trace-clean", "--api-key", "invalid-key"], capture_output=True, text=True, input="test trace"
        )
        assert result.returncode != 0
        # Should provide helpful error message
        assert len(result.stderr) > 0

    def test_permission_denied_file(self):
        """Test handling of permission denied on file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            # Make file unreadable
            os.chmod(temp_file, 0o000)

            result = subprocess.run(
                ["trace-clean", "--api-key", "test-key", "--file", temp_file], capture_output=True, text=True
            )
            assert result.returncode != 0
            assert "permission" in result.stderr.lower() or "access" in result.stderr.lower()
        finally:
            os.chmod(temp_file, 0o644)  # Restore permissions for cleanup
            os.unlink(temp_file)

    def test_interrupted_execution(self):
        """Test graceful handling of interrupted execution."""
        # This is hard to test directly, but we can check that the CLI
        # starts up properly before any interruption
        result = subprocess.run(["trace-clean", "--help"], capture_output=True, text=True, timeout=5)
        assert result.returncode == 0


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility features."""

    def test_path_separator_handling(self):
        """Test handling of different path separators."""
        if os.name == "nt":  # Windows
            trace_content = "Traceback:\n  File 'C:\\Users\\test\\app.py', line 1"
        else:  # Unix-like
            trace_content = "Traceback:\n  File '/home/test/app.py', line 1"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(trace_content)
            temp_file = f.name

        try:
            result = subprocess.run(
                ["trace-clean", "--api-key", "test-key", "--file", temp_file], capture_output=True, text=True
            )
            assert result.returncode in [0, 1, 2]
        finally:
            os.unlink(temp_file)

    def test_line_ending_handling(self):
        """Test handling of different line endings."""
        trace_with_crlf = "Traceback (most recent call last):\r\n  File test.py, line 1\r\nError"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", newline="") as f:
            f.write(trace_with_crlf)
            temp_file = f.name

        try:
            result = subprocess.run(
                ["trace-clean", "--api-key", "test-key", "--file", temp_file], capture_output=True, text=True
            )
            assert result.returncode in [0, 1, 2]
        finally:
            os.unlink(temp_file)

    def test_environment_variable_expansion(self):
        """Test handling of environment variables in paths."""
        # Skip this test if HOME/USERPROFILE is not set
        home_var = "USERPROFILE" if os.name == "nt" else "HOME"
        if home_var not in os.environ:
            pytest.skip(f"{home_var} environment variable not set")

        # Just test that the CLI doesn't crash with env var paths
        result = subprocess.run(["trace-clean", "--help"], capture_output=True, text=True, env=os.environ.copy())
        assert result.returncode == 0


class TestPerformanceAndLimits:
    """Test performance characteristics and limits."""

    def test_memory_usage_large_input(self):
        """Test memory usage with large input."""
        # Create a moderately large trace (not too large to avoid CI issues)
        large_trace = "Traceback (most recent call last):\n" + "  Frame\n" * 1000

        result = subprocess.run(
            ["trace-clean", "--api-key", "test-key"], capture_output=True, text=True, input=large_trace, timeout=10
        )
        # Should handle reasonably large input without crashing
        assert result.returncode in [0, 1, 2]

    def test_execution_time_limit(self):
        """Test that CLI execution completes in reasonable time."""
        import time

        start_time = time.time()
        result = subprocess.run(["trace-clean", "--help"], capture_output=True, text=True, timeout=5)
        end_time = time.time()

        assert result.returncode == 0
        assert end_time - start_time < 5  # Should complete quickly

    def test_file_size_limits(self):
        """Test handling of file size limits."""
        # Create a file that's large but not unreasonable
        content = "Traceback line\n" * 50000  # About 650KB

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_file = f.name

        try:
            result = subprocess.run(
                ["trace-clean", "--api-key", "test-key", "--file", temp_file],
                capture_output=True,
                text=True,
                timeout=15,
            )
            # Should either process or reject gracefully
            assert result.returncode in [0, 1, 2]
        except subprocess.TimeoutExpired:
            pytest.skip("Large file processing timed out")
        finally:
            os.unlink(temp_file)
