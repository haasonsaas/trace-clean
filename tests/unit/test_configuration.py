"""
Unit tests for configuration functionality.
This module contains 22 comprehensive tests covering all configuration functionality.
"""

import json
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
import pytest

from trace_clean_pkg import TraceClean


class TestConfiguration:
    """Test suite for configuration functionality - 22 tests total."""
    
    def test_default_config_values(self):
        """Test default configuration values are set correctly."""
        with patch.dict(os.environ, {}, clear=True):
            trace_clean = TraceClean(model="local")
            
            # Test default values in config
            assert trace_clean.model == "local"
            assert trace_clean.local_url == "http://localhost:11434"
            assert isinstance(trace_clean.config, dict)  # Config loaded (may have defaults)
    
    def test_config_file_loading(self):
        """Test configuration file loading from ~/.trace-clean/config.yaml."""
        mock_config = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "timeout": 60,
            "local_model": "custom-llama",
            "local_url": "http://custom:8080"
        }
        
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('builtins.open', mock_open(read_data="temperature: 0.7\nmax_tokens: 1000")) as mock_file:
                with patch('yaml.safe_load', return_value=mock_config):
                    trace_clean = TraceClean(model="local")
                    
                    # Verify config was loaded
                    assert trace_clean.config == mock_config
                    assert trace_clean.local_url == "http://custom:8080"
                    
                    # Verify file operations
                    expected_path = Path.home() / ".trace-clean" / "config.yaml"
                    mock_exists.assert_called_with()
                    mock_file.assert_called_once()
    
    def test_config_all_mentioned_options(self):
        """Test all configuration options mentioned in requirements."""
        config_options = {
            "temperature": 0.5,
            "max_tokens": 2000,
            "timeout": 120,
            "local_model": "llama3.2",
            "local_url": "http://localhost:9999"
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open()):
                with patch('yaml.safe_load', return_value=config_options):
                    trace_clean = TraceClean(model="local")
                    
                    # Verify all options are in config
                    assert trace_clean.config["temperature"] == 0.5
                    assert trace_clean.config["max_tokens"] == 2000
                    assert trace_clean.config["timeout"] == 120
                    assert trace_clean.config["local_model"] == "llama3.2"
                    assert trace_clean.config["local_url"] == "http://localhost:9999"
                    
                    # Verify local_url was applied
                    assert trace_clean.local_url == "http://localhost:9999"
    
    def test_environment_variable_precedence(self):
        """Test environment variables take precedence over config file."""
        config_data = {
            "temperature": 0.3,
            "max_tokens": 1000,
            "local_url": "http://config-file:8080"
        }
        
        # Test API key from environment
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-api-key'}):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open()):
                    with patch('yaml.safe_load', return_value=config_data):
                        trace_clean = TraceClean(model="gpt-4o-mini")
                        
                        # Environment API key should be used
                        assert trace_clean.api_key == "env-api-key"
                        
                        # Config file values should still be loaded
                        assert trace_clean.config["temperature"] == 0.3
                        assert trace_clean.config["max_tokens"] == 1000
    
    def test_command_line_flag_precedence(self):
        """Test command line flags take precedence over config and environment."""
        config_data = {
            "local_url": "http://config:8080"
        }
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open()):
                    with patch('yaml.safe_load', return_value=config_data):
                        # Explicit parameters should override everything
                        trace_clean = TraceClean(
                            model="gpt-4",
                            api_key="explicit-key",
                            local_url="http://explicit:9000"
                        )
                        
                        # Explicit values should be used
                        assert trace_clean.model == "gpt-4"
                        assert trace_clean.api_key == "explicit-key"
                        assert trace_clean.local_url == "http://explicit:9000"
                        
                        # Config should still be loaded for other values
                        assert "local_url" in trace_clean.config
    
    def test_invalid_config_file_handling(self):
        """Test handling of invalid/corrupted config files."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="invalid: yaml: content: [")):
                with patch('yaml.safe_load', side_effect=yaml.YAMLError("Invalid YAML")):
                    with patch('trace_clean_pkg.cli.logger.warning') as mock_warning:
                        trace_clean = TraceClean(model="local")
                        
                        # Should use empty config on error
                        assert trace_clean.config == {}
                        
                        # Should log warning
                        mock_warning.assert_called_once()
                        assert "Failed to load config" in str(mock_warning.call_args[0][0])
    
    def test_config_file_with_permission_error(self):
        """Test handling when config file exists but can't be read (permissions)."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                with patch('trace_clean_pkg.cli.logger.warning') as mock_warning:
                    trace_clean = TraceClean(model="local")
                    
                    # Should use empty config on permission error
                    assert trace_clean.config == {}
                    
                    # Should log warning
                    mock_warning.assert_called_once()
                    assert "Failed to load config" in str(mock_warning.call_args[0][0])
    
    def test_config_directory_creation(self):
        """Test config directory path resolution."""
        expected_path = Path.home() / ".trace-clean" / "config.yaml"
        
        with patch('pathlib.Path.exists', return_value=False):
            trace_clean = TraceClean(model="local")
            
            # Even though config doesn't exist, path calculation should work
            # This is tested indirectly through the _load_config method
            assert trace_clean.config == {}
    
    def test_yaml_parsing_errors(self):
        """Test various YAML parsing error scenarios."""
        error_scenarios = [
            ("invalid: yaml: [", yaml.YAMLError("Invalid syntax")),
            ("", yaml.YAMLError("Empty file")),
            ("key: value\n  invalid_indent", yaml.YAMLError("Indentation error")),
        ]
        
        for yaml_content, error in error_scenarios:
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open(read_data=yaml_content)):
                    with patch('yaml.safe_load', side_effect=error):
                        with patch('trace_clean_pkg.cli.logger.warning') as mock_warning:
                            trace_clean = TraceClean(model="local")
                            
                            assert trace_clean.config == {}
                            mock_warning.assert_called_once()
    
    def test_empty_config_file(self):
        """Test handling of empty config file."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="")):
                with patch('yaml.safe_load', return_value=None):
                    trace_clean = TraceClean(model="local")
                    
                    # Empty config should result in empty dict
                    assert trace_clean.config == {}
                    assert trace_clean.local_url == "http://localhost:11434"  # Default value
    
    def test_config_with_null_values(self):
        """Test handling of config with null/None values."""
        config_with_nulls = {
            "temperature": None,
            "max_tokens": 1500,
            "timeout": None,
            "local_model": "llama3",
            "local_url": None
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open()):
                with patch('yaml.safe_load', return_value=config_with_nulls):
                    trace_clean = TraceClean(model="local")
                    
                    # Null values should be preserved in config
                    assert trace_clean.config["temperature"] is None
                    assert trace_clean.config["max_tokens"] == 1500
                    assert trace_clean.config["timeout"] is None
                    assert trace_clean.config["local_model"] == "llama3"
                    assert trace_clean.config["local_url"] is None
                    
                    # local_url should not be updated if None in config
                    assert trace_clean.local_url == "http://localhost:11434"  # Default
    
    def test_api_key_validation(self):
        """Test API key validation in various scenarios."""
        # Test missing API key for OpenAI models
        with pytest.raises(ValueError, match="API key required"):
            TraceClean(model="gpt-4o-mini")
        
        with pytest.raises(ValueError, match="API key required"):
            TraceClean(model="gpt-4")
        
        # Test API key not required for local model
        trace_clean = TraceClean(model="local")
        assert trace_clean.model == "local"
        
        # Test API key validation with config file
        config_data = {"temperature": 0.5}
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open()):
                with patch('yaml.safe_load', return_value=config_data):
                    with pytest.raises(ValueError, match="API key required"):
                        TraceClean(model="gpt-4o-mini")
    
    def test_local_url_validation(self):
        """Test local URL validation in config scenarios."""
        # Test invalid URL in config
        config_with_invalid_url = {
            "local_url": "invalid-url-format"
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open()):
                with patch('yaml.safe_load', return_value=config_with_invalid_url):
                    with pytest.raises(ValueError, match="Invalid URL format"):
                        TraceClean(model="local")
        
        # Test valid URL in config
        config_with_valid_url = {
            "local_url": "http://valid:8080"
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open()):
                with patch('yaml.safe_load', return_value=config_with_valid_url):
                    trace_clean = TraceClean(model="local")
                    assert trace_clean.local_url == "http://valid:8080"
    
    def test_config_values_used_in_api_calls(self):
        """Test configuration values are used in OpenAI API calls."""
        config_data = {
            "temperature": 0.8,
            "max_tokens": 2500
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open()):
                with patch('yaml.safe_load', return_value=config_data):
                    trace_clean = TraceClean(model="gpt-4o-mini", api_key="test-key")
                    
                    # Mock OpenAI client
                    with patch('openai.OpenAI') as mock_openai:
                        mock_client = Mock()
                        mock_response = Mock()
                        mock_response.choices = [Mock()]
                        mock_response.choices[0].message.content = '{"test": "response"}'
                        mock_client.chat.completions.create.return_value = mock_response
                        mock_openai.return_value = mock_client
                        
                        # Test API call uses config values
                        result = trace_clean._call_openai("test prompt")
                        
                        # Verify config values were used
                        call_args = mock_client.chat.completions.create.call_args[1]
                        assert call_args["temperature"] == 0.8
                        assert call_args["max_tokens"] == 2500
    
    def test_config_values_used_in_local_calls(self):
        """Test configuration values are used in local model calls."""
        config_data = {
            "temperature": 0.2,
            "max_tokens": 3000,
            "timeout": 180,
            "local_model": "custom-model"
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open()):
                with patch('yaml.safe_load', return_value=config_data):
                    trace_clean = TraceClean(model="local")
                    
                    # Mock requests for local API
                    with patch('requests.post') as mock_post:
                        mock_response = Mock()
                        mock_response.json.return_value = {"response": '{"test": "response"}'}
                        mock_response.raise_for_status = Mock()
                        mock_post.return_value = mock_response
                        
                        # Test local API call uses config values
                        result = trace_clean._call_local_model("test prompt")
                        
                        # Verify config values were used
                        call_args = mock_post.call_args
                        payload = call_args[1]["json"]
                        assert payload["model"] == "custom-model"
                        assert payload["temperature"] == 0.2
                        assert payload["options"]["num_predict"] == 3000
                        
                        # Verify timeout was used
                        assert call_args[1]["timeout"] == 180
    
    def test_config_explicit_vs_implicit_local_url(self):
        """Test explicit vs implicit local URL handling."""
        config_data = {
            "local_url": "http://config-file:8080"
        }
        
        # Test implicit local_url (not provided in constructor)
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open()):
                with patch('yaml.safe_load', return_value=config_data):
                    trace_clean = TraceClean(model="local")
                    # Should use config file value
                    assert trace_clean.local_url == "http://config-file:8080"
        
        # Test explicit local_url (provided in constructor)
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open()):
                with patch('yaml.safe_load', return_value=config_data):
                    trace_clean = TraceClean(model="local", local_url="http://explicit:9000")
                    # Should use explicit value, not config file
                    assert trace_clean.local_url == "http://explicit:9000"
    
    def test_config_loading_file_not_found(self):
        """Test config loading when file doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            trace_clean = TraceClean(model="local")
            
            # Should use empty config and default values
            assert trace_clean.config == {}
            assert trace_clean.local_url == "http://localhost:11434"
    
    def test_config_partial_values(self):
        """Test config with only some values specified."""
        partial_config = {
            "temperature": 0.4,
            # Missing max_tokens, timeout, etc.
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open()):
                with patch('yaml.safe_load', return_value=partial_config):
                    trace_clean = TraceClean(model="local")
                    
                    # Should have the specified value
                    assert trace_clean.config["temperature"] == 0.4
                    
                    # Should use defaults for unspecified values in actual calls
                    with patch('requests.post') as mock_post:
                        mock_response = Mock()
                        mock_response.json.return_value = {"response": '{"test": "response"}'}
                        mock_response.raise_for_status = Mock()
                        mock_post.return_value = mock_response
                        
                        trace_clean._call_local_model("test")
                        
                        payload = mock_post.call_args[1]["json"]
                        assert payload["temperature"] == 0.4  # From config
                        assert payload["options"]["num_predict"] == 2000  # Default
                        assert mock_post.call_args[1]["timeout"] == 120  # Default
    
    def test_config_environment_interaction(self):
        """Test interaction between config file and environment variables."""
        config_data = {
            "temperature": 0.6,
            "local_url": "http://config:8080"
        }
        
        # Test with and without environment API key
        test_cases = [
            ({}, None),  # No env var
            ({"OPENAI_API_KEY": "env-key"}, "env-key"),  # With env var
            ({"OPENAI_API_KEY": ""}, ""),  # Empty env var
        ]
        
        for env_vars, expected_api_key in test_cases:
            with patch.dict(os.environ, env_vars, clear=True):
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('builtins.open', mock_open()):
                        with patch('yaml.safe_load', return_value=config_data):
                            if expected_api_key:
                                trace_clean = TraceClean(model="gpt-4o-mini")
                                assert trace_clean.api_key == expected_api_key
                            else:
                                # Should raise error if no API key and non-local model
                                with pytest.raises(ValueError, match="API key required"):
                                    TraceClean(model="gpt-4o-mini")
                            
                            # Local model should always work regardless of API key
                            trace_clean_local = TraceClean(model="local")
                            assert trace_clean_local.config["temperature"] == 0.6
                            assert trace_clean_local.local_url == "http://config:8080"
    
    def test_config_edge_cases_and_defaults(self):
        """Test configuration edge cases and default value fallbacks."""
        edge_cases = [
            # Case 1: Config exists but is empty dict
            {},
            # Case 2: Config has some invalid types
            {"temperature": "invalid", "max_tokens": "also_invalid", "timeout": 90},
            # Case 3: Config has extra unexpected keys
            {"temperature": 0.5, "unexpected_key": "value", "another_key": 123},
        ]
        
        for config_data in edge_cases:
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open()):
                    with patch('yaml.safe_load', return_value=config_data):
                        trace_clean = TraceClean(model="local")
                        
                        # Should not crash with any config data
                        assert trace_clean.config == config_data
                        
                        # Test that get() with defaults works
                        temp = trace_clean.config.get("temperature", 0.3)
                        max_tokens = trace_clean.config.get("max_tokens", 1500)
                        timeout = trace_clean.config.get("timeout", 120)
                        
                        # Values should be from config or defaults
                        if "temperature" in config_data:
                            assert temp == config_data["temperature"]
                        else:
                            assert temp == 0.3
                        
                        if "timeout" in config_data:
                            assert timeout == config_data["timeout"]
                        else:
                            assert timeout == 120
    
    def test_config_complex_integration_scenario(self):
        """Test complex integration scenario with config, env vars, and explicit params."""
        config_data = {
            "temperature": 0.7,
            "max_tokens": 1800,
            "timeout": 150,
            "local_model": "config-model",
            "local_url": "http://config:8080"
        }
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"}):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open()):
                    with patch('yaml.safe_load', return_value=config_data):
                        # Create with mixed explicit and implicit values
                        trace_clean = TraceClean(
                            model="gpt-4o-mini",  # explicit
                            local_url="http://explicit:9000"  # explicit, should override config
                            # api_key not provided, should come from environment
                        )
                        
                        # Verify precedence: explicit > env > config > default
                        assert trace_clean.model == "gpt-4o-mini"  # explicit
                        assert trace_clean.api_key == "env-api-key"  # environment
                        assert trace_clean.local_url == "http://explicit:9000"  # explicit override
                        assert trace_clean.config["temperature"] == 0.7  # config
                        assert trace_clean.config["max_tokens"] == 1800  # config
                        assert trace_clean.config["timeout"] == 150  # config
    
    def test_config_file_path_resolution(self):
        """Test configuration file path resolution and access patterns."""
        # Test path construction
        expected_path = Path.home() / ".trace-clean" / "config.yaml"
        
        # Test various path scenarios
        with patch('pathlib.Path.home') as mock_home:
            mock_home.return_value = Path("/custom/home")
            
            # Mock the path operations
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open(read_data="temperature: 0.5")):
                    with patch('yaml.safe_load', return_value={"temperature": 0.5}):
                        trace_clean = TraceClean(model="local")
                        
                        # Should still load config successfully
                        assert trace_clean.config["temperature"] == 0.5
                        
                        # Verify home() was called for path resolution
                        mock_home.assert_called()