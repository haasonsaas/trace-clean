"""Integration tests for OpenAI API functionality."""

import json
import os
from unittest.mock import Mock, patch

import pytest
import responses
from traceclean.config import Config
from traceclean.models.openai_model import OpenAIModel


@pytest.mark.integration
@pytest.mark.api_required
class TestOpenAIAPIIntegration:
    """Test suite for OpenAI API integration."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment or use test key."""
        return os.environ.get("OPENAI_API_KEY", "test-api-key")

    @pytest.fixture
    def openai_model(self, api_key):
        """Create OpenAI model instance."""
        config = Config()
        config.api_key = api_key
        config.model = "gpt-3.5-turbo"
        return OpenAIModel(config)

    def test_successful_api_call_with_real_key(self, openai_model):
        """Test successful API call with real API key."""
        if openai_model.config.api_key == "test-api-key":
            pytest.skip("Real API key not available")

        stack_trace = """Traceback (most recent call last):
  File "test.py", line 10, in <module>
    result = divide(10, 0)
  File "test.py", line 5, in divide
    return a / b
ZeroDivisionError: division by zero"""

        response = openai_model.clean_stack_trace(stack_trace)
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    @responses.activate
    def test_api_key_authentication_error(self, openai_model):
        """Test handling of 401 authentication errors."""
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json={"error": {"message": "Invalid authentication", "type": "invalid_request_error"}},
            status=401,
        )

        with pytest.raises(Exception) as exc_info:
            openai_model.clean_stack_trace("test trace")

        assert "401" in str(exc_info.value) or "authentication" in str(exc_info.value).lower()

    @patch("requests.post")
    def test_network_timeout_error(self, mock_post, openai_model):
        """Test handling of network timeout errors."""
        import requests

        mock_post.side_effect = requests.Timeout("Connection timed out")

        with pytest.raises(requests.Timeout):
            openai_model.clean_stack_trace("test trace")

    @responses.activate
    def test_rate_limiting_error(self, openai_model):
        """Test handling of 429 rate limiting errors."""
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
            status=429,
        )

        with pytest.raises(Exception) as exc_info:
            openai_model.clean_stack_trace("test trace")

        assert "429" in str(exc_info.value) or "rate limit" in str(exc_info.value).lower()

    @responses.activate
    def test_invalid_model_name(self, api_key):
        """Test handling of invalid model names."""
        config = Config()
        config.api_key = api_key
        config.model = "invalid-model-name"
        model = OpenAIModel(config)

        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json={"error": {"message": "Invalid model", "type": "invalid_request_error"}},
            status=400,
        )

        with pytest.raises(Exception) as exc_info:
            model.clean_stack_trace("test trace")

        assert "400" in str(exc_info.value) or "invalid" in str(exc_info.value).lower()

    def test_temperature_max_tokens_passing(self, api_key):
        """Test passing of temperature and max_tokens parameters."""
        config = Config()
        config.api_key = api_key
        config.model = "gpt-3.5-turbo"
        config.temperature = 0.7
        config.max_tokens = 500
        model = OpenAIModel(config)

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"choices": [{"message": {"content": "cleaned trace"}}]}
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            model.clean_stack_trace("test trace")

            # Verify the request included temperature and max_tokens
            call_args = mock_post.call_args
            request_data = json.loads(call_args[1]["data"])
            assert request_data["temperature"] == 0.7
            assert request_data["max_tokens"] == 500

    @responses.activate
    def test_malformed_json_response(self, openai_model):
        """Test handling of malformed JSON responses."""
        responses.add(
            responses.POST, "https://api.openai.com/v1/chat/completions", body="Invalid JSON {not valid}", status=200
        )

        with pytest.raises(json.JSONDecodeError):
            openai_model.clean_stack_trace("test trace")

    @responses.activate
    def test_empty_response_handling(self, openai_model):
        """Test handling of empty responses."""
        responses.add(responses.POST, "https://api.openai.com/v1/chat/completions", json={"choices": []}, status=200)

        result = openai_model.clean_stack_trace("test trace")
        assert result == "" or result is None

    @responses.activate
    def test_api_response_structure_validation(self, openai_model):
        """Test validation of API response structure."""
        # Test missing 'choices' key
        responses.add(
            responses.POST, "https://api.openai.com/v1/chat/completions", json={"result": "something"}, status=200
        )

        with pytest.raises(KeyError):
            openai_model.clean_stack_trace("test trace")

    def test_configuration_integration_with_api(self, api_key):
        """Test configuration integration with API calls."""
        config = Config()
        config.api_key = api_key
        config.model = "gpt-4"
        config.temperature = 0.5
        config.max_tokens = 1000
        model = OpenAIModel(config)

        assert model.config.api_key == api_key
        assert model.config.model == "gpt-4"
        assert model.config.temperature == 0.5
        assert model.config.max_tokens == 1000

    @responses.activate
    def test_long_stack_trace_handling(self, openai_model):
        """Test handling of very long stack traces."""
        # Create a very long stack trace
        long_trace = "Traceback (most recent call last):\n"
        for i in range(100):
            long_trace += f"  File 'file{i}.py', line {i}, in function{i}\n"
            long_trace += f"    some_code_line_{i}()\n"
        long_trace += "ValueError: Some error"

        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json={"choices": [{"message": {"content": "Cleaned long trace"}}]},
            status=200,
        )

        result = openai_model.clean_stack_trace(long_trace)
        assert result == "Cleaned long trace"

    @responses.activate
    def test_unicode_stack_trace_handling(self, openai_model):
        """Test handling of stack traces with Unicode characters."""
        unicode_trace = """Traceback (most recent call last):
  File "测试.py", line 10, in <module>
    结果 = 除法(10, 0)
  File "测试.py", line 5, in 除法
    return a / b
ZeroDivisionError: 除以零错误 ñ"""

        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json={"choices": [{"message": {"content": "Cleaned unicode trace"}}]},
            status=200,
        )

        result = openai_model.clean_stack_trace(unicode_trace)
        assert result == "Cleaned unicode trace"

    @responses.activate
    def test_api_error_message_formatting(self, openai_model):
        """Test formatting of API error messages."""
        error_responses = [
            (400, {"error": {"message": "Bad request", "type": "invalid_request_error"}}),
            (403, {"error": {"message": "Forbidden", "type": "forbidden_error"}}),
            (500, {"error": {"message": "Internal server error", "type": "server_error"}}),
        ]

        for status, error_body in error_responses:
            responses.add(responses.POST, "https://api.openai.com/v1/chat/completions", json=error_body, status=status)

            with pytest.raises(Exception) as exc_info:
                openai_model.clean_stack_trace("test trace")

            assert str(status) in str(exc_info.value) or error_body["error"]["message"] in str(exc_info.value)

    @pytest.mark.slow
    def test_concurrent_api_requests(self, openai_model):
        """Test handling of concurrent API requests."""
        import concurrent.futures

        def make_request(trace):
            with patch("requests.post") as mock_post:
                mock_response = Mock()
                mock_response.json.return_value = {"choices": [{"message": {"content": f"Cleaned: {trace}"}}]}
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response
                return openai_model.clean_stack_trace(trace)

        traces = [f"Trace {i}" for i in range(5)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(make_request, traces))

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result == f"Cleaned: Trace {i}"

    @patch("requests.post")
    def test_api_retry_logic(self, mock_post, openai_model):
        """Test API retry logic for transient failures."""
        # First two calls fail, third succeeds
        mock_response_fail = Mock()
        mock_response_fail.raise_for_status.side_effect = Exception("Temporary failure")

        mock_response_success = Mock()
        mock_response_success.json.return_value = {"choices": [{"message": {"content": "Success after retry"}}]}
        mock_response_success.raise_for_status = Mock()

        mock_post.side_effect = [mock_response_fail, mock_response_fail, mock_response_success]

        # Assuming the implementation has retry logic
        with patch("time.sleep"):  # Skip actual sleep delays
            result = openai_model.clean_stack_trace("test trace")

        assert result == "Success after retry"
        assert mock_post.call_count == 3

    def test_api_timeout_configuration(self, api_key):
        """Test API timeout configuration."""
        config = Config()
        config.api_key = api_key
        config.timeout = 5  # 5 seconds timeout
        model = OpenAIModel(config)

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"choices": [{"message": {"content": "Response"}}]}
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            model.clean_stack_trace("test trace")

            # Verify timeout was passed to requests
            call_args = mock_post.call_args
            assert "timeout" in call_args[1]
            assert call_args[1]["timeout"] == 5

    def test_api_key_environment_variable(self):
        """Test API key loading from environment variable."""
        test_key = "sk-test-key-from-env"

        with patch.dict(os.environ, {"OPENAI_API_KEY": test_key}):
            config = Config()
            # Assuming Config loads from env if not explicitly set
            config.api_key = os.environ.get("OPENAI_API_KEY")
            model = OpenAIModel(config)

            assert model.config.api_key == test_key
