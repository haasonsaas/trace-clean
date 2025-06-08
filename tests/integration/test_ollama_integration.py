"""Integration tests for Ollama functionality."""

import json
from unittest.mock import Mock, patch

import pytest
import requests
from traceclean.config import Config
from traceclean.models.ollama_model import OllamaModel


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestOllamaModelDetection:
    """Test suite for Ollama model detection functionality."""

    @pytest.fixture
    def config(self):
        """Create basic config for Ollama."""
        config = Config()
        config.model = "llama2"
        config.ollama_host = "http://localhost:11434"
        return config

    @pytest.fixture
    def ollama_model(self, config):
        """Create Ollama model instance."""
        return OllamaModel(config)

    @patch("requests.get")
    def test_ollama_server_available_detection(self, mock_get, ollama_model):
        """Test detection of available Ollama server."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "0.1.0"}
        mock_get.return_value = mock_response

        available = ollama_model.is_server_available()
        assert available is True

    @patch("requests.get")
    def test_ollama_server_unavailable_detection(self, mock_get, ollama_model):
        """Test detection of unavailable Ollama server."""
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        available = ollama_model.is_server_available()
        assert available is False

    @patch("requests.get")
    def test_available_models_list_retrieval(self, mock_get, ollama_model):
        """Test retrieval of available models list."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2:latest", "size": 3826793677},
                {"name": "codellama:latest", "size": 3826793677},
                {"name": "mistral:latest", "size": 4109016676},
            ]
        }
        mock_get.return_value = mock_response

        models = ollama_model.get_available_models()
        assert len(models) == 3
        assert "llama2:latest" in models
        assert "codellama:latest" in models
        assert "mistral:latest" in models

    @patch("requests.get")
    def test_model_exists_check_positive(self, mock_get, ollama_model):
        """Test checking if a specific model exists (positive case)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama2:latest", "size": 3826793677}, {"name": "codellama:latest", "size": 3826793677}]
        }
        mock_get.return_value = mock_response

        exists = ollama_model.model_exists("llama2")
        assert exists is True

    @patch("requests.get")
    def test_model_exists_check_negative(self, mock_get, ollama_model):
        """Test checking if a specific model exists (negative case)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama2:latest", "size": 3826793677}, {"name": "codellama:latest", "size": 3826793677}]
        }
        mock_get.return_value = mock_response

        exists = ollama_model.model_exists("nonexistent-model")
        assert exists is False


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestOllamaConnection:
    """Test suite for Ollama connection functionality."""

    @pytest.fixture
    def config(self):
        """Create basic config for Ollama."""
        config = Config()
        config.model = "llama2"
        config.ollama_host = "http://localhost:11434"
        return config

    @pytest.fixture
    def ollama_model(self, config):
        """Create Ollama model instance."""
        return OllamaModel(config)

    @patch("requests.get")
    def test_connection_success_with_default_host(self, mock_get, ollama_model):
        """Test successful connection to default Ollama host."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "0.1.0"}
        mock_get.return_value = mock_response

        connected = ollama_model.test_connection()
        assert connected is True
        mock_get.assert_called_once_with("http://localhost:11434/api/version", timeout=5)

    @patch("requests.get")
    def test_connection_success_with_custom_host(self, mock_get, config):
        """Test successful connection to custom Ollama host."""
        config.ollama_host = "http://192.168.1.100:11434"
        ollama_model = OllamaModel(config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "0.1.0"}
        mock_get.return_value = mock_response

        connected = ollama_model.test_connection()
        assert connected is True
        mock_get.assert_called_once_with("http://192.168.1.100:11434/api/version", timeout=5)

    @patch("requests.get")
    def test_connection_failure_timeout(self, mock_get, ollama_model):
        """Test connection failure due to timeout."""
        mock_get.side_effect = requests.Timeout("Request timed out")

        connected = ollama_model.test_connection()
        assert connected is False

    @patch("requests.get")
    def test_connection_failure_connection_error(self, mock_get, ollama_model):
        """Test connection failure due to connection error."""
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        connected = ollama_model.test_connection()
        assert connected is False

    @patch("requests.get")
    def test_connection_failure_http_error(self, mock_get, ollama_model):
        """Test connection failure due to HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        connected = ollama_model.test_connection()
        assert connected is False


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestOllamaModelSelection:
    """Test suite for Ollama model selection functionality."""

    @pytest.fixture
    def config(self):
        """Create basic config for Ollama."""
        config = Config()
        config.ollama_host = "http://localhost:11434"
        return config

    def test_default_model_selection(self, config):
        """Test default model selection when none specified."""
        config.model = None
        ollama_model = OllamaModel(config)

        # Should default to a common model
        assert ollama_model.config.model in ["llama2", "llama2:latest", "mistral", "mistral:latest"]

    def test_explicit_model_selection(self, config):
        """Test explicit model selection."""
        config.model = "codellama"
        ollama_model = OllamaModel(config)

        assert ollama_model.config.model == "codellama"

    @patch("requests.get")
    def test_model_validation_against_available_models(self, mock_get, config):
        """Test model validation against available models."""
        config.model = "llama2"
        ollama_model = OllamaModel(config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama2:latest", "size": 3826793677}, {"name": "codellama:latest", "size": 3826793677}]
        }
        mock_get.return_value = mock_response

        is_valid = ollama_model.validate_model()
        assert is_valid is True


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestOllamaResponseParsing:
    """Test suite for Ollama response parsing functionality."""

    @pytest.fixture
    def config(self):
        """Create basic config for Ollama."""
        config = Config()
        config.model = "llama2"
        config.ollama_host = "http://localhost:11434"
        return config

    @pytest.fixture
    def ollama_model(self, config):
        """Create Ollama model instance."""
        return OllamaModel(config)

    def test_successful_response_parsing(self, ollama_model):
        """Test parsing of successful Ollama response."""
        response_data = {
            "model": "llama2",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "response": "The cleaned stack trace is:\n\nZeroDivisionError: division by zero",
            "done": True,
        }

        result = ollama_model.parse_response(response_data)
        assert result == "The cleaned stack trace is:\n\nZeroDivisionError: division by zero"

    def test_streaming_response_parsing(self, ollama_model):
        """Test parsing of streaming Ollama responses."""
        streaming_responses = [
            {"response": "The ", "done": False},
            {"response": "cleaned ", "done": False},
            {"response": "stack trace", "done": False},
            {"response": "", "done": True},
        ]

        result = ollama_model.parse_streaming_responses(streaming_responses)
        assert result == "The cleaned stack trace"

    def test_empty_response_handling(self, ollama_model):
        """Test handling of empty responses."""
        response_data = {"model": "llama2", "created_at": "2023-08-04T19:22:45.499127Z", "response": "", "done": True}

        result = ollama_model.parse_response(response_data)
        assert result == ""

    def test_malformed_response_handling(self, ollama_model):
        """Test handling of malformed responses."""
        malformed_response = {"invalid": "structure"}

        with pytest.raises(KeyError):
            ollama_model.parse_response(malformed_response)


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestOllamaTimeoutHandling:
    """Test suite for Ollama timeout handling functionality."""

    @pytest.fixture
    def config(self):
        """Create basic config for Ollama."""
        config = Config()
        config.model = "llama2"
        config.ollama_host = "http://localhost:11434"
        config.timeout = 10
        return config

    @pytest.fixture
    def ollama_model(self, config):
        """Create Ollama model instance."""
        return OllamaModel(config)

    @patch("requests.post")
    def test_request_timeout_configuration(self, mock_post, ollama_model):
        """Test that request timeout is properly configured."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Cleaned trace", "done": True}
        mock_post.return_value = mock_response

        ollama_model.clean_stack_trace("test trace")

        # Verify timeout was passed to requests
        call_args = mock_post.call_args
        assert "timeout" in call_args[1]
        assert call_args[1]["timeout"] == 10

    @patch("requests.post")
    def test_timeout_exception_handling(self, mock_post, ollama_model):
        """Test handling of timeout exceptions."""
        mock_post.side_effect = requests.Timeout("Request timed out")

        with pytest.raises(requests.Timeout):
            ollama_model.clean_stack_trace("test trace")

    @patch("requests.post")
    def test_long_running_request_timeout(self, mock_post, config):
        """Test timeout for long-running requests."""
        config.timeout = 1  # Very short timeout
        ollama_model = OllamaModel(config)

        # Simulate a slow response
        def slow_response(*args, **kwargs):
            import time

            time.sleep(2)  # Longer than timeout
            return Mock()

        mock_post.side_effect = slow_response

        with pytest.raises(requests.Timeout):
            ollama_model.clean_stack_trace("test trace")

    def test_default_timeout_configuration(self, config):
        """Test default timeout configuration."""
        config.timeout = None
        ollama_model = OllamaModel(config)

        # Should have a reasonable default timeout
        default_timeout = getattr(ollama_model.config, "timeout", 30)
        assert default_timeout > 0
        assert default_timeout <= 60  # Reasonable upper bound


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestOllamaErrorHandling:
    """Test suite for Ollama error handling functionality."""

    @pytest.fixture
    def config(self):
        """Create basic config for Ollama."""
        config = Config()
        config.model = "llama2"
        config.ollama_host = "http://localhost:11434"
        return config

    @pytest.fixture
    def ollama_model(self, config):
        """Create Ollama model instance."""
        return OllamaModel(config)

    @patch("requests.post")
    def test_model_not_found_error(self, mock_post, config):
        """Test handling of model not found errors."""
        config.model = "nonexistent-model"
        ollama_model = OllamaModel(config)

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "model 'nonexistent-model' not found"}
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as exc_info:
            ollama_model.clean_stack_trace("test trace")

        assert "not found" in str(exc_info.value).lower()

    @patch("requests.post")
    def test_server_error_handling(self, mock_post, ollama_model):
        """Test handling of server errors (5xx)."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error"}
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_post.return_value = mock_response

        with pytest.raises(requests.HTTPError):
            ollama_model.clean_stack_trace("test trace")

    @patch("requests.post")
    def test_network_connectivity_error(self, mock_post, ollama_model):
        """Test handling of network connectivity errors."""
        mock_post.side_effect = requests.ConnectionError("Network is unreachable")

        with pytest.raises(requests.ConnectionError):
            ollama_model.clean_stack_trace("test trace")

    @patch("requests.post")
    def test_invalid_json_response_error(self, mock_post, ollama_model):
        """Test handling of invalid JSON responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_response

        with pytest.raises(json.JSONDecodeError):
            ollama_model.clean_stack_trace("test trace")

    @patch("requests.post")
    def test_insufficient_memory_error(self, mock_post, ollama_model):
        """Test handling of insufficient memory errors."""
        mock_response = Mock()
        mock_response.status_code = 413
        mock_response.json.return_value = {"error": "insufficient memory to load model"}
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as exc_info:
            ollama_model.clean_stack_trace("test trace")

        assert "memory" in str(exc_info.value).lower()

    @patch("requests.post")
    def test_rate_limiting_error(self, mock_post, ollama_model):
        """Test handling of rate limiting errors."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": "rate limit exceeded"}
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as exc_info:
            ollama_model.clean_stack_trace("test trace")

        assert "rate limit" in str(exc_info.value).lower()


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestOllamaConfigurationIntegration:
    """Test suite for Ollama configuration integration."""

    def test_host_configuration_from_config(self):
        """Test host configuration from config object."""
        config = Config()
        config.ollama_host = "http://custom-host:11434"
        config.model = "llama2"

        ollama_model = OllamaModel(config)
        assert ollama_model.config.ollama_host == "http://custom-host:11434"

    def test_host_configuration_with_port(self):
        """Test host configuration with custom port."""
        config = Config()
        config.ollama_host = "http://localhost:8080"
        config.model = "llama2"

        ollama_model = OllamaModel(config)
        assert ollama_model.config.ollama_host == "http://localhost:8080"

    def test_model_configuration_persistence(self):
        """Test that model configuration persists through operations."""
        config = Config()
        config.model = "codellama"
        config.ollama_host = "http://localhost:11434"

        ollama_model = OllamaModel(config)

        # Configuration should persist
        assert ollama_model.config.model == "codellama"

        # After operations, config should still be the same
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "result", "done": True}
            mock_post.return_value = mock_response

            ollama_model.clean_stack_trace("test")
            assert ollama_model.config.model == "codellama"

    def test_temperature_and_options_configuration(self):
        """Test temperature and other options configuration."""
        config = Config()
        config.model = "llama2"
        config.temperature = 0.7
        config.top_p = 0.9
        config.top_k = 40

        ollama_model = OllamaModel(config)

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "result", "done": True}
            mock_post.return_value = mock_response

            ollama_model.clean_stack_trace("test")

            # Verify options were passed in request
            call_args = mock_post.call_args
            request_data = json.loads(call_args[1]["data"])

            assert "options" in request_data
            options = request_data["options"]
            assert options.get("temperature") == 0.7
            assert options.get("top_p") == 0.9
            assert options.get("top_k") == 40

    def test_system_prompt_configuration(self):
        """Test system prompt configuration."""
        config = Config()
        config.model = "llama2"
        config.system_prompt = "You are a helpful assistant for cleaning stack traces."

        ollama_model = OllamaModel(config)

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "result", "done": True}
            mock_post.return_value = mock_response

            ollama_model.clean_stack_trace("test")

            # Verify system prompt was included
            call_args = mock_post.call_args
            request_data = json.loads(call_args[1]["data"])

            assert "system" in request_data
            assert request_data["system"] == "You are a helpful assistant for cleaning stack traces."


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestOllamaRealWorldScenarios:
    """Test suite for real-world Ollama usage scenarios."""

    @pytest.fixture
    def config(self):
        """Create realistic config for Ollama."""
        config = Config()
        config.model = "llama2"
        config.ollama_host = "http://localhost:11434"
        config.temperature = 0.1  # Low temperature for consistent output
        return config

    @pytest.fixture
    def ollama_model(self, config):
        """Create Ollama model instance."""
        return OllamaModel(config)

    @patch("requests.post")
    def test_python_stack_trace_cleaning(self, mock_post, ollama_model):
        """Test cleaning of realistic Python stack trace."""
        python_trace = """Traceback (most recent call last):
  File "/Users/developer/project/main.py", line 45, in <module>
    result = process_data(data)
  File "/Users/developer/project/utils.py", line 123, in process_data
    cleaned = clean_data(raw_data)
  File "/Users/developer/project/utils.py", line 89, in clean_data
    return data.strip().upper()
AttributeError: 'NoneType' object has no attribute 'strip'"""

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "AttributeError: 'NoneType' object has no attribute 'strip'\nCaused by: data is None in clean_data function",
            "done": True,
        }
        mock_post.return_value = mock_response

        result = ollama_model.clean_stack_trace(python_trace)
        assert "AttributeError" in result
        assert "NoneType" in result

    @patch("requests.post")
    def test_javascript_stack_trace_cleaning(self, mock_post, ollama_model):
        """Test cleaning of JavaScript stack trace."""
        js_trace = """TypeError: Cannot read property 'length' of undefined
    at processArray (/app/src/utils.js:42:18)
    at handleData (/app/src/controller.js:89:12)
    at /app/src/app.js:156:9
    at Layer.handle [as handle_request] (/app/node_modules/express/lib/router/layer.js:95:5)
    at next (/app/node_modules/express/lib/router/route.js:137:13)"""

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "TypeError: Cannot read property 'length' of undefined\nCaused by: array is undefined in processArray function",
            "done": True,
        }
        mock_post.return_value = mock_response

        result = ollama_model.clean_stack_trace(js_trace)
        assert "TypeError" in result
        assert "undefined" in result

    @patch("requests.post")
    def test_concurrent_trace_processing(self, mock_post, ollama_model):
        """Test processing multiple traces concurrently."""
        import concurrent.futures

        traces = [
            "ValueError: invalid literal for int()",
            "KeyError: 'missing_key'",
            "IndexError: list index out of range",
        ]

        def mock_response_generator(trace_content):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": f"Cleaned: {trace_content}", "done": True}
            return mock_response

        mock_post.side_effect = [mock_response_generator(trace) for trace in traces]

        def process_trace(trace):
            return ollama_model.clean_stack_trace(trace)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(process_trace, traces))

        assert len(results) == 3
        for i, result in enumerate(results):
            assert f"Cleaned: {traces[i]}" in result


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestOllamaAdvancedFeatures:
    """Test suite for advanced Ollama features."""

    @pytest.fixture
    def config(self):
        """Create config with advanced features."""
        config = Config()
        config.model = "llama2"
        config.ollama_host = "http://localhost:11434"
        return config

    @pytest.fixture
    def ollama_model(self, config):
        """Create Ollama model instance."""
        return OllamaModel(config)

    @patch("requests.post")
    def test_streaming_response_processing(self, mock_post, ollama_model):
        """Test processing of streaming responses."""
        # Mock streaming response
        streaming_data = [
            '{"response": "The ", "done": false}',
            '{"response": "error ", "done": false}',
            '{"response": "is: ", "done": false}',
            '{"response": "ValueError", "done": true}',
        ]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = streaming_data
        mock_post.return_value = mock_response

        # Assuming the model supports streaming
        result = ollama_model.clean_stack_trace_streaming("test trace")
        assert "The error is: ValueError" in result

    @patch("requests.post")
    def test_context_window_management(self, mock_post, ollama_model):
        """Test management of context window for large inputs."""
        # Create a very large stack trace that might exceed context window
        large_trace = "Traceback (most recent call last):\n"
        for i in range(1000):  # Very large trace
            large_trace += f"  File 'file{i}.py', line {i}, in function{i}\n"
            large_trace += f"    some_very_long_line_of_code_that_might_exceed_limits_{i}()\n"
        large_trace += "ValueError: Some error"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Truncated and cleaned trace", "done": True}
        mock_post.return_value = mock_response

        result = ollama_model.clean_stack_trace(large_trace)
        assert result == "Truncated and cleaned trace"

    @patch("requests.post")
    def test_model_performance_metrics(self, mock_post, ollama_model):
        """Test collection of model performance metrics."""
        import time

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Cleaned trace",
            "done": True,
            "total_duration": 1500000000,  # nanoseconds
            "load_duration": 500000000,
            "prompt_eval_count": 50,
            "eval_count": 25,
        }
        mock_post.return_value = mock_response

        start_time = time.time()
        result = ollama_model.clean_stack_trace("test trace")
        end_time = time.time()

        # Verify result and timing
        assert result == "Cleaned trace"
        assert (end_time - start_time) < 5  # Should complete quickly in test

    @patch("requests.post")
    def test_custom_prompt_templates(self, mock_post, config):
        """Test custom prompt templates for different trace types."""
        config.prompt_template = "Clean this {language} stack trace: {trace}"
        ollama_model = OllamaModel(config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Custom cleaned trace", "done": True}
        mock_post.return_value = mock_response

        ollama_model.clean_stack_trace("test trace", language="Python")

        # Verify custom prompt was used
        call_args = mock_post.call_args
        request_data = json.loads(call_args[1]["data"])
        assert "Clean this Python stack trace: test trace" in request_data["prompt"]

    @patch("requests.post")
    def test_model_switching_support(self, mock_post, config):
        """Test support for switching models dynamically."""
        # Start with one model
        config.model = "llama2"
        ollama_model = OllamaModel(config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Llama2 response", "done": True}
        mock_post.return_value = mock_response

        result1 = ollama_model.clean_stack_trace("test trace")
        assert result1 == "Llama2 response"

        # Switch to different model
        ollama_model.switch_model("codellama")

        mock_response.json.return_value = {"response": "CodeLlama response", "done": True}

        result2 = ollama_model.clean_stack_trace("test trace")
        assert result2 == "CodeLlama response"

        # Verify model was switched in requests
        calls = mock_post.call_args_list
        assert len(calls) == 2

        request1_data = json.loads(calls[0][1]["data"])
        request2_data = json.loads(calls[1][1]["data"])

        assert request1_data["model"] == "llama2"
        assert request2_data["model"] == "codellama"
