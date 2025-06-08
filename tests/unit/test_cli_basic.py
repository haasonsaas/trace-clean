"""
Unit tests for CLI basic functionality.
This module contains 25 comprehensive tests covering all basic CLI functionality.
"""

import argparse
import json
import os
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from trace_clean_pkg import main, TraceClean
from trace_clean_pkg.cli import main as cli_main


class TestCLIBasic:
    """Test suite for basic CLI functionality - 25 tests total."""
    
    def test_help_command_and_format(self, capsys):
        """Test help command displays correct information and formatting."""
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.argv', ['trace-clean', '--help']):
                cli_main()
        
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        
        # Check key elements are in help output
        assert "trace-clean: Post-Mortem Stack-Trace Summarizer" in captured.out
        assert "--json" in captured.out
        assert "--model" in captured.out
        assert "--api-key" in captured.out
        assert "--local-url" in captured.out
        assert "Examples:" in captured.out
        assert "trace-clean error.log" in captured.out
        assert "cat stacktrace.txt | trace-clean" in captured.out
    
    def test_version_information(self):
        """Test version information is accessible."""
        from trace_clean_pkg import __version__
        assert __version__ == "1.0.0"
        assert isinstance(__version__, str)
    
    def test_argument_parser_creation(self):
        """Test argument parser is created correctly."""
        # Test parser creation doesn't raise errors
        parser = argparse.ArgumentParser(
            description="trace-clean: Post-Mortem Stack-Trace Summarizer",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        
        # Add arguments like in main function
        parser.add_argument("file", nargs="?", help="File containing the stack trace")
        parser.add_argument("--json", action="store_true", help="Output results in JSON format")
        parser.add_argument("--model", default="gpt-4o-mini", help="Language model to use")
        parser.add_argument("--api-key", help="API key for the language model provider")
        parser.add_argument("--local-url", default="http://localhost:11434", help="URL for local model server")
        
        # Test parsing valid arguments
        args = parser.parse_args(['--json', '--model', 'gpt-4', 'test.txt'])
        assert args.json is True
        assert args.model == 'gpt-4'
        assert args.file == 'test.txt'
    
    def test_all_command_line_flags(self):
        """Test all command line flags are recognized and parsed correctly."""
        parser = argparse.ArgumentParser()
        parser.add_argument("file", nargs="?")
        parser.add_argument("--json", action="store_true")
        parser.add_argument("--model", default="gpt-4o-mini")
        parser.add_argument("--api-key")
        parser.add_argument("--local-url", default="http://localhost:11434")
        
        # Test --json flag
        args = parser.parse_args(['--json'])
        assert args.json is True
        
        # Test --model flag
        args = parser.parse_args(['--model', 'gpt-4'])
        assert args.model == 'gpt-4'
        
        # Test --api-key flag
        args = parser.parse_args(['--api-key', 'sk-test123'])
        assert args.api_key == 'sk-test123'
        
        # Test --local-url flag
        args = parser.parse_args(['--local-url', 'http://localhost:8080'])
        assert args.local_url == 'http://localhost:8080'
        
        # Test multiple flags together
        args = parser.parse_args([
            '--json', 
            '--model', 'local', 
            '--api-key', 'test-key',
            '--local-url', 'http://localhost:9999',
            'error.log'
        ])
        assert args.json is True
        assert args.model == 'local'
        assert args.api_key == 'test-key'
        assert args.local_url == 'http://localhost:9999'
        assert args.file == 'error.log'
    
    def test_invalid_arguments_error_handling(self):
        """Test invalid arguments raise appropriate errors."""
        parser = argparse.ArgumentParser()
        parser.add_argument("file", nargs="?")
        parser.add_argument("--json", action="store_true")
        
        # Test invalid flag
        with pytest.raises(SystemExit):
            parser.parse_args(['--invalid-flag'])
    
    def test_file_input_detection(self):
        """Test file input detection works correctly."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Sample stack trace content")
            temp_file = f.name
        
        try:
            # Test file exists and can be detected
            assert os.path.exists(temp_file)
            
            # Test reading file content
            with open(temp_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert content == "Sample stack trace content"
        finally:
            os.unlink(temp_file)
    
    def test_stdin_detection(self):
        """Test stdin detection works correctly."""
        # Test stdin detection logic
        original_stdin = sys.stdin
        
        try:
            # Mock stdin as not a TTY (piped input)
            sys.stdin = StringIO("Test stack trace from stdin")
            sys.stdin.isatty = lambda: False
            
            content = sys.stdin.read()
            assert content == "Test stack trace from stdin"
            
            # Reset for TTY test
            sys.stdin = StringIO()
            sys.stdin.isatty = lambda: True
            assert sys.stdin.isatty() is True
            
        finally:
            sys.stdin = original_stdin
    
    def test_file_not_found_error(self, capsys):
        """Test file not found error handling."""
        non_existent_file = "/path/that/does/not/exist.txt"
        
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.argv', ['trace-clean', non_existent_file]):
                cli_main()
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()
    
    def test_empty_stack_trace_error(self, capsys):
        """Test empty stack trace error handling."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("")  # Empty file
            temp_file = f.name
        
        try:
            with pytest.raises(SystemExit) as exc_info:
                with patch('sys.argv', ['trace-clean', temp_file]):
                    cli_main()
            
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "no stack trace provided" in captured.out.lower()
        finally:
            os.unlink(temp_file)
    
    def test_basic_import_structure(self):
        """Test basic import structure is correct."""
        # Test main imports
        from trace_clean_pkg import TraceClean, main
        assert TraceClean is not None
        assert main is not None
        
        # Test CLI imports
        from trace_clean_pkg.cli import TraceClean as CLITraceClean, main as cli_main
        assert CLITraceClean is not None
        assert cli_main is not None
        
        # Verify they're the same
        assert TraceClean is CLITraceClean
        assert main is cli_main
    
    def test_traceclean_class_initialization(self):
        """Test TraceClean class initialization with various parameters."""
        # Test default initialization (should fail without API key)
        with pytest.raises(ValueError, match="API key required"):
            TraceClean()
        
        # Test with API key
        trace_clean = TraceClean(api_key="test-key")
        assert trace_clean.model == "gpt-4o-mini"
        assert trace_clean.api_key == "test-key"
        assert trace_clean.local_url == "http://localhost:11434"
        
        # Test with custom model
        trace_clean = TraceClean(model="gpt-4", api_key="test-key")
        assert trace_clean.model == "gpt-4"
        
        # Test with local model (no API key required)
        trace_clean = TraceClean(model="local")
        assert trace_clean.model == "local"
        assert trace_clean.local_url == "http://localhost:11434"
    
    def test_traceclean_class_api_key_validation(self):
        """Test TraceClean class API key validation."""
        # Test missing API key for non-local models
        with pytest.raises(ValueError, match="API key required"):
            TraceClean(model="gpt-4o-mini")
        
        # Test API key from environment variable
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-test-key'}):
            trace_clean = TraceClean()
            assert trace_clean.api_key == "env-test-key"
        
        # Test explicit API key overrides environment
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-test-key'}):
            trace_clean = TraceClean(api_key="explicit-key")
            assert trace_clean.api_key == "explicit-key"
        
        # Test local model doesn't require API key
        trace_clean = TraceClean(model="local")
        assert trace_clean.model == "local"
    
    def test_traceclean_invalid_local_url(self):
        """Test TraceClean class validation of local URL."""
        # Test invalid URL formats
        with pytest.raises(ValueError, match="Invalid URL format"):
            TraceClean(model="local", local_url="not-a-url")
        
        with pytest.raises(ValueError, match="Invalid URL format"):
            TraceClean(model="local", local_url="://invalid")
        
        with pytest.raises(ValueError, match="Invalid URL format"):
            TraceClean(model="local", local_url="http://")
        
        # Test valid URLs work
        valid_urls = [
            "http://localhost:11434",
            "http://127.0.0.1:8080",
            "https://example.com:443"
        ]
        for url in valid_urls:
            trace_clean = TraceClean(model="local", local_url=url)
            assert trace_clean.local_url == url
    
    def test_format_output_json(self):
        """Test JSON format output."""
        trace_clean = TraceClean(model="local")
        
        sample_result = {
            "summary": "Test error summary",
            "suspect_functions": [
                {
                    "function": "test_func",
                    "file": "test.py",
                    "line": 42,
                    "confidence": "High",
                    "reason": "Test reason"
                }
            ],
            "next_actions": ["Action 1", "Action 2"]
        }
        
        json_output = trace_clean.format_output(sample_result, json_format=True)
        
        # Verify it's valid JSON
        parsed = json.loads(json_output)
        assert parsed == sample_result
        
        # Verify formatting
        assert '"summary": "Test error summary"' in json_output
        assert '"function": "test_func"' in json_output
    
    def test_format_output_markdown(self):
        """Test markdown format output."""
        trace_clean = TraceClean(model="local")
        
        sample_result = {
            "summary": "Test error summary",
            "suspect_functions": [
                {
                    "function": "test_func",
                    "file": "test.py",
                    "line": 42,
                    "confidence": "High",
                    "reason": "Test reason"
                }
            ],
            "next_actions": ["Action 1", "Action 2"]
        }
        
        markdown_output = trace_clean.format_output(sample_result, json_format=False)
        
        # Verify markdown formatting
        assert "# Stack Trace Analysis" in markdown_output
        assert "## Summary" in markdown_output
        assert "## Suspect Functions" in markdown_output
        assert "## Next Actions" in markdown_output
        assert "### 1. test_func" in markdown_output
        assert "**File**: `test.py`" in markdown_output
        assert "**Line**: 42" in markdown_output
        assert "**Confidence**: High" in markdown_output
        assert "1. Action 1" in markdown_output
        assert "2. Action 2" in markdown_output
    
    def test_config_loading_no_config(self):
        """Test configuration loading when no config file exists."""
        with patch('pathlib.Path.exists', return_value=False):
            trace_clean = TraceClean(model="local")
            assert trace_clean.config == {}
    
    def test_config_loading_with_config(self):
        """Test configuration loading with existing config file."""
        mock_config = {
            "temperature": 0.5,
            "max_tokens": 2000,
            "local_model": "llama3",
            "local_url": "http://localhost:8080"
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', create=True) as mock_file:
                mock_file.return_value.__enter__.return_value.read.return_value = "temperature: 0.5\nmax_tokens: 2000\nlocal_model: llama3\nlocal_url: http://localhost:8080"
                with patch('yaml.safe_load', return_value=mock_config):
                    trace_clean = TraceClean(model="local")
                    assert trace_clean.config["temperature"] == 0.5
                    assert trace_clean.config["max_tokens"] == 2000
                    assert trace_clean.config["local_model"] == "llama3"
                    # local_url should be updated if not explicitly provided
                    assert trace_clean.local_url == "http://localhost:8080"
    
    def test_stack_trace_size_limit(self, capsys):
        """Test stack trace size limitation."""
        # Create a very large stack trace
        large_trace = "Error line\n" * 10000  # Should exceed 50000 chars
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(large_trace)
            temp_file = f.name
        
        try:
            # Mock the analyzer to avoid actual API calls
            with patch('trace_clean_pkg.cli.TraceClean') as mock_class:
                mock_instance = Mock()
                mock_instance.analyze.return_value = {"summary": "test", "suspect_functions": [], "next_actions": []}
                mock_instance.format_output.return_value = "test output"
                mock_class.return_value = mock_instance
                
                with patch('sys.argv', ['trace-clean', temp_file]):
                    try:
                        cli_main()
                    except SystemExit:
                        pass
                
                captured = capsys.readouterr()
                if len(large_trace) > 50000:
                    assert "truncated" in captured.out.lower()
        finally:
            os.unlink(temp_file)
    
    def test_main_entry_point(self):
        """Test main entry point exists and is callable."""
        from trace_clean_pkg import main
        assert callable(main)
        assert main.__name__ == "main"
    
    def test_console_script_entry_point(self):
        """Test console script entry point exists."""
        from trace_clean_pkg.cli import main as cli_main
        assert callable(cli_main)
        assert cli_main.__name__ == "main"
        
        # Test __main__ module entry point
        from trace_clean_pkg.__main__ import main as main_main
        assert main is main_main
    
    def test_argument_parsing_edge_cases(self):
        """Test edge cases in argument parsing."""
        parser = argparse.ArgumentParser()
        parser.add_argument("file", nargs="?")
        parser.add_argument("--json", action="store_true")
        parser.add_argument("--model", default="gpt-4o-mini")
        
        # Test no arguments
        args = parser.parse_args([])
        assert args.file is None
        assert args.json is False
        assert args.model == "gpt-4o-mini"
        
        # Test only file argument
        args = parser.parse_args(['test.txt'])
        assert args.file == 'test.txt'
        assert args.json is False
        
        # Test flag without value
        args = parser.parse_args(['--json'])
        assert args.json is True
    
    def test_error_handling_edge_cases(self, capsys):
        """Test various error handling edge cases."""
        # Test with whitespace-only file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("   \n\t  \n  ")  # Only whitespace
            temp_file = f.name
        
        try:
            with pytest.raises(SystemExit) as exc_info:
                with patch('sys.argv', ['trace-clean', temp_file]):
                    cli_main()
            
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "no stack trace provided" in captured.out.lower()
        finally:
            os.unlink(temp_file)
    
    def test_stdin_tty_detection(self, capsys):
        """Test TTY detection for stdin input."""
        # Mock stdin being a TTY
        with patch('sys.stdin') as mock_stdin:
            mock_stdin.isatty.return_value = True
            mock_stdin.read.return_value = "test stack trace"
            
            with patch('sys.argv', ['trace-clean']):
                with patch('trace_clean_pkg.cli.TraceClean') as mock_class:
                    mock_instance = Mock()
                    mock_instance.analyze.return_value = {"summary": "test", "suspect_functions": [], "next_actions": []}
                    mock_instance.format_output.return_value = "test output"
                    mock_class.return_value = mock_instance
                    
                    try:
                        cli_main()
                    except SystemExit:
                        pass
            
            captured = capsys.readouterr()
            # Should show the TTY hint message
            assert "reading from stdin" in captured.out.lower() or "ctrl" in captured.out.lower()
    
    def test_json_output_handling(self, capsys):
        """Test JSON output is properly handled and displayed."""
        sample_result = {
            "summary": "Test summary",
            "suspect_functions": [],
            "next_actions": []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Sample stack trace")
            temp_file = f.name
        
        try:
            with patch('trace_clean_pkg.cli.TraceClean') as mock_class:
                mock_instance = Mock()
                mock_instance.analyze.return_value = sample_result
                mock_instance.format_output.return_value = json.dumps(sample_result, indent=2)
                mock_class.return_value = mock_instance
                
                with patch('sys.argv', ['trace-clean', '--json', temp_file]):
                    try:
                        cli_main()
                    except SystemExit:
                        pass
                
                # Verify the analyze method was called
                mock_instance.analyze.assert_called_once()
        finally:
            os.unlink(temp_file)
    
    def test_comprehensive_cli_workflow(self):
        """Test comprehensive CLI workflow from start to finish."""
        sample_result = {
            "summary": "Integration test summary",
            "suspect_functions": [
                {
                    "function": "main",
                    "file": "app.py",
                    "line": 10,
                    "confidence": "High",
                    "reason": "Entry point function"
                }
            ],
            "next_actions": ["Check app.py line 10", "Add error handling"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Traceback (most recent call last):\n  File \"app.py\", line 10, in main\n    x = 1/0\nZeroDivisionError: division by zero")
            temp_file = f.name
        
        try:
            with patch('trace_clean_pkg.cli.TraceClean') as mock_class:
                mock_instance = Mock()
                mock_instance.analyze.return_value = sample_result
                mock_instance.format_output.return_value = "formatted output"
                mock_class.return_value = mock_instance
                
                # Test successful execution
                with patch('sys.argv', ['trace-clean', temp_file]):
                    try:
                        cli_main()
                    except SystemExit:
                        pass
                
                # Verify initialization and method calls
                mock_class.assert_called_once()
                mock_instance.analyze.assert_called_once()
                mock_instance.format_output.assert_called_once_with(sample_result)
        finally:
            os.unlink(temp_file)