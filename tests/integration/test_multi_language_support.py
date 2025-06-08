"""
Integration tests for multi-language support in trace-clean.
Tests various programming languages, error types, and API responses.
"""
import pytest
from unittest.mock import Mock, patch
import json
from trace_clean_pkg.trace_cleaner import TraceCleaner
from trace_clean_pkg.openai_client import OpenAIClient


class TestMultiLanguageSupport:
    """Test trace cleaning across multiple programming languages."""
    
    @pytest.fixture
    def cleaner(self):
        """Create a TraceCleaner instance for testing."""
        return TraceCleaner(api_key="test-key")
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        client = Mock(spec=OpenAIClient)
        return client
    
    # Python Language Tests (Tests 1-5)
    @pytest.mark.parametrize("trace_type", ["KeyError", "ValueError", "TypeError", "AttributeError", "IndexError"])
    def test_python_error_types(self, cleaner, mock_client, trace_type):
        """Test different Python error types are handled correctly."""
        python_trace = f"""
Traceback (most recent call last):
  File "app.py", line 45, in process_data
    result = data['missing_key']
{trace_type}: 'missing_key'
        """
        
        mock_response = {
            "cleaned_trace": f"Python {trace_type} in process_data function",
            "suspect_functions": ["process_data"],
            "confidence": "High",
            "next_actions": [f"Check for {trace_type} handling"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(python_trace)
            
        assert result["confidence"] == "High"
        assert "process_data" in result["suspect_functions"]
        assert trace_type in str(result["cleaned_trace"])
    
    # Java Language Tests (Tests 6-9)
    def test_java_null_pointer_exception(self, cleaner, mock_client):
        """Test Java NullPointerException handling."""
        java_trace = """
Exception in thread "main" java.lang.NullPointerException
    at com.example.UserService.getUser(UserService.java:42)
    at com.example.Main.main(Main.java:15)
        """
        
        mock_response = {
            "cleaned_trace": "NullPointerException in UserService.getUser",
            "suspect_functions": ["getUser", "main"],
            "confidence": "High",
            "next_actions": ["Check null handling in getUser method"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(java_trace)
            
        assert result["confidence"] == "High"
        assert "getUser" in result["suspect_functions"]
    
    def test_java_class_not_found(self, cleaner, mock_client):
        """Test Java ClassNotFoundException."""
        java_trace = """
java.lang.ClassNotFoundException: com.example.MissingClass
    at java.net.URLClassLoader.findClass(URLClassLoader.java:381)
    at java.lang.ClassLoader.loadClass(ClassLoader.java:424)
        """
        
        mock_response = {
            "cleaned_trace": "ClassNotFoundException: com.example.MissingClass",
            "suspect_functions": ["loadClass"],
            "confidence": "Medium",
            "next_actions": ["Verify classpath configuration"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(java_trace)
            
        assert result["confidence"] == "Medium"
        assert "Verify classpath" in str(result["next_actions"])
    
    def test_java_array_index_out_of_bounds(self, cleaner, mock_client):
        """Test Java ArrayIndexOutOfBoundsException."""
        java_trace = """
java.lang.ArrayIndexOutOfBoundsException: Index 10 out of bounds for length 5
    at com.example.DataProcessor.processArray(DataProcessor.java:78)
        """
        
        mock_response = {
            "cleaned_trace": "Array index 10 exceeds array length 5",
            "suspect_functions": ["processArray"],
            "confidence": "High",
            "next_actions": ["Add bounds checking in processArray"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(java_trace)
            
        assert "processArray" in result["suspect_functions"]
    
    def test_java_illegal_argument(self, cleaner, mock_client):
        """Test Java IllegalArgumentException."""
        java_trace = """
java.lang.IllegalArgumentException: Invalid parameter value: -1
    at com.example.Validator.validate(Validator.java:23)
        """
        
        mock_response = {
            "cleaned_trace": "Invalid parameter: negative value not allowed",
            "suspect_functions": ["validate"],
            "confidence": "High",
            "next_actions": ["Add parameter validation"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(java_trace)
            
        assert result["confidence"] == "High"
    
    # JavaScript Language Tests (Tests 10-13)
    def test_javascript_type_error(self, cleaner, mock_client):
        """Test JavaScript TypeError handling."""
        js_trace = """
TypeError: Cannot read property 'name' of undefined
    at Object.getUserName (/app/user.js:15:20)
    at processRequest (/app/server.js:45:10)
        """
        
        mock_response = {
            "cleaned_trace": "Attempting to access 'name' on undefined object",
            "suspect_functions": ["getUserName", "processRequest"],
            "confidence": "High",
            "next_actions": ["Add null checks before property access"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(js_trace)
            
        assert "getUserName" in result["suspect_functions"]
    
    def test_javascript_reference_error(self, cleaner, mock_client):
        """Test JavaScript ReferenceError."""
        js_trace = """
ReferenceError: userData is not defined
    at calculateTotal (/app/calculator.js:23:5)
        """
        
        mock_response = {
            "cleaned_trace": "Variable 'userData' not defined",
            "suspect_functions": ["calculateTotal"],
            "confidence": "High",
            "next_actions": ["Define userData before use"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(js_trace)
            
        assert result["suspect_functions"] == ["calculateTotal"]
    
    def test_javascript_syntax_error(self, cleaner, mock_client):
        """Test JavaScript SyntaxError."""
        js_trace = """
SyntaxError: Unexpected token '}'
    at Module._compile (internal/modules/cjs/loader.js:723:23)
        """
        
        mock_response = {
            "cleaned_trace": "Syntax error: unexpected closing brace",
            "suspect_functions": [],
            "confidence": "Medium",
            "next_actions": ["Check for matching braces"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(js_trace)
            
        assert result["confidence"] == "Medium"
    
    def test_javascript_promise_rejection(self, cleaner, mock_client):
        """Test JavaScript unhandled promise rejection."""
        js_trace = """
UnhandledPromiseRejectionWarning: Error: Connection timeout
    at DatabaseConnection.connect (/app/db.js:34:11)
        """
        
        mock_response = {
            "cleaned_trace": "Database connection timeout",
            "suspect_functions": ["connect"],
            "confidence": "High",
            "next_actions": ["Add promise error handling"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(js_trace)
            
        assert "connect" in result["suspect_functions"]
    
    # Ruby Language Tests (Tests 14-16)
    def test_ruby_no_method_error(self, cleaner, mock_client):
        """Test Ruby NoMethodError."""
        ruby_trace = """
NoMethodError: undefined method `fetch' for nil:NilClass
    from app/models/user.rb:25:in `get_profile'
    from app/controllers/users_controller.rb:10:in `show'
        """
        
        mock_response = {
            "cleaned_trace": "Calling fetch on nil object",
            "suspect_functions": ["get_profile", "show"],
            "confidence": "High",
            "next_actions": ["Check for nil before method call"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(ruby_trace)
            
        assert len(result["suspect_functions"]) == 2
    
    def test_ruby_argument_error(self, cleaner, mock_client):
        """Test Ruby ArgumentError."""
        ruby_trace = """
ArgumentError: wrong number of arguments (given 2, expected 1)
    from lib/calculator.rb:15:in `calculate'
        """
        
        mock_response = {
            "cleaned_trace": "Method called with 2 arguments, expects 1",
            "suspect_functions": ["calculate"],
            "confidence": "High",
            "next_actions": ["Fix method call arguments"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(ruby_trace)
            
        assert result["confidence"] == "High"
    
    def test_ruby_load_error(self, cleaner, mock_client):
        """Test Ruby LoadError."""
        ruby_trace = """
LoadError: cannot load such file -- missing_gem
    from /usr/lib/ruby/2.7.0/rubygems/core_ext/kernel_require.rb:92:in `require'
        """
        
        mock_response = {
            "cleaned_trace": "Missing dependency: missing_gem",
            "suspect_functions": ["require"],
            "confidence": "Medium",
            "next_actions": ["Install missing gem"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(ruby_trace)
            
        assert "Install missing gem" in result["next_actions"]
    
    # Go Language Tests (Tests 17-19)
    def test_go_panic_nil_pointer(self, cleaner, mock_client):
        """Test Go panic with nil pointer."""
        go_trace = """
panic: runtime error: invalid memory address or nil pointer dereference
[signal SIGSEGV: segmentation violation code=0x1 addr=0x0 pc=0x4a1b2f]

goroutine 1 [running]:
main.processUser(0x0)
    /app/main.go:45 +0x2f
        """
        
        mock_response = {
            "cleaned_trace": "Nil pointer dereference in processUser",
            "suspect_functions": ["processUser"],
            "confidence": "High",
            "next_actions": ["Add nil checks in processUser"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(go_trace)
            
        assert "processUser" in result["suspect_functions"]
    
    def test_go_index_out_of_range(self, cleaner, mock_client):
        """Test Go index out of range."""
        go_trace = """
panic: runtime error: index out of range [5] with length 3

goroutine 1 [running]:
main.getElement(...)
    /app/array.go:23
        """
        
        mock_response = {
            "cleaned_trace": "Array index 5 exceeds length 3",
            "suspect_functions": ["getElement"],
            "confidence": "High",
            "next_actions": ["Add bounds checking"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(go_trace)
            
        assert result["confidence"] == "High"
    
    def test_go_deadlock(self, cleaner, mock_client):
        """Test Go deadlock detection."""
        go_trace = """
fatal error: all goroutines are asleep - deadlock!

goroutine 1 [chan send]:
main.main()
    /app/deadlock.go:15 +0x5c
        """
        
        mock_response = {
            "cleaned_trace": "Deadlock detected: all goroutines blocked",
            "suspect_functions": ["main"],
            "confidence": "High",
            "next_actions": ["Review channel operations"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(go_trace)
            
        assert "Review channel operations" in result["next_actions"]
    
    # Rust Language Tests (Tests 20-22)
    def test_rust_panic(self, cleaner, mock_client):
        """Test Rust panic handling."""
        rust_trace = """
thread 'main' panicked at 'index out of bounds: the len is 3 but the index is 5', src/main.rs:10:5
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
        """
        
        mock_response = {
            "cleaned_trace": "Index 5 out of bounds for array of length 3",
            "suspect_functions": ["main"],
            "confidence": "High",
            "next_actions": ["Use safe indexing methods"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(rust_trace)
            
        assert result["confidence"] == "High"
    
    def test_rust_unwrap_none(self, cleaner, mock_client):
        """Test Rust unwrap on None."""
        rust_trace = """
thread 'main' panicked at 'called `Option::unwrap()` on a `None` value', src/user.rs:25:14
        """
        
        mock_response = {
            "cleaned_trace": "Unwrap called on None value",
            "suspect_functions": [],
            "confidence": "High",
            "next_actions": ["Use match or if let instead of unwrap"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(rust_trace)
            
        assert "Use match" in str(result["next_actions"])
    
    def test_rust_borrow_checker_error(self, cleaner, mock_client):
        """Test Rust borrow checker error."""
        rust_trace = """
error[E0382]: use of moved value: `data`
  --> src/main.rs:15:20
   |
14 |     let result = process(data);
   |                          ---- value moved here
15 |     println!("{}", data);
   |                    ^^^^ value used here after move
        """
        
        mock_response = {
            "cleaned_trace": "Value used after move",
            "suspect_functions": ["process"],
            "confidence": "High",
            "next_actions": ["Clone data or use reference"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(rust_trace)
            
        assert "Clone data" in str(result["next_actions"])
    
    # C++ Language Tests (Tests 23-24)
    def test_cpp_segmentation_fault(self, cleaner, mock_client):
        """Test C++ segmentation fault."""
        cpp_trace = """
Program received signal SIGSEGV, Segmentation fault.
0x0000000000400546 in processArray (arr=0x0) at main.cpp:15
15          return arr[0];
        """
        
        mock_response = {
            "cleaned_trace": "Segmentation fault: null pointer access",
            "suspect_functions": ["processArray"],
            "confidence": "High",
            "next_actions": ["Check for null before access"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(cpp_trace)
            
        assert "processArray" in result["suspect_functions"]
    
    def test_cpp_std_exception(self, cleaner, mock_client):
        """Test C++ std::exception."""
        cpp_trace = """
terminate called after throwing an instance of 'std::out_of_range'
  what():  vector::_M_range_check: __n (which is 10) >= this->size() (which is 5)
Aborted (core dumped)
        """
        
        mock_response = {
            "cleaned_trace": "Vector index 10 out of range (size: 5)",
            "suspect_functions": [],
            "confidence": "High",
            "next_actions": ["Use at() with try-catch or check size"]
        }
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = mock_response
            result = cleaner.clean_trace(cpp_trace)
            
        assert result["confidence"] == "High"
    
    # Edge Cases and Validation Tests (Tests 25-30)
    def test_empty_trace(self, cleaner, mock_client):
        """Test handling of empty trace."""
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = {
                "cleaned_trace": "No trace provided",
                "suspect_functions": [],
                "confidence": "Low",
                "next_actions": ["Provide stack trace"]
            }
            result = cleaner.clean_trace("")
            
        assert result["confidence"] == "Low"
    
    def test_malformed_trace(self, cleaner, mock_client):
        """Test handling of malformed trace."""
        malformed = "This is not a valid stack trace at all!"
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = {
                "cleaned_trace": "Unable to parse trace format",
                "suspect_functions": [],
                "confidence": "Low",
                "next_actions": ["Provide valid stack trace"]
            }
            result = cleaner.clean_trace(malformed)
            
        assert result["confidence"] == "Low"
    
    def test_very_long_trace(self, cleaner, mock_client):
        """Test handling of very long traces."""
        long_trace = "\n".join([f"at function{i} (file.py:{i})" for i in range(1000)])
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = {
                "cleaned_trace": "Stack overflow detected",
                "suspect_functions": ["function999"],
                "confidence": "Medium",
                "next_actions": ["Check for infinite recursion"]
            }
            result = cleaner.clean_trace(long_trace)
            
        assert "Check for infinite recursion" in result["next_actions"]
    
    def test_mixed_language_trace(self, cleaner, mock_client):
        """Test trace with mixed language indicators."""
        mixed_trace = """
Traceback (most recent call last):
Exception in thread "main" java.lang.NullPointerException
TypeError: Cannot read property 'name' of undefined
        """
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = {
                "cleaned_trace": "Multiple errors detected across languages",
                "suspect_functions": [],
                "confidence": "Low",
                "next_actions": ["Separate traces by language"]
            }
            result = cleaner.clean_trace(mixed_trace)
            
        assert result["confidence"] == "Low"
    
    def test_confidence_levels(self, cleaner, mock_client):
        """Test all confidence levels are handled."""
        traces = [
            ("High confidence trace", "High"),
            ("Medium confidence trace", "Medium"),
            ("Low confidence trace", "Low")
        ]
        
        for trace, confidence in traces:
            with patch.object(cleaner, 'client', mock_client):
                mock_client.clean_trace.return_value = {
                    "cleaned_trace": trace,
                    "suspect_functions": [],
                    "confidence": confidence,
                    "next_actions": []
                }
                result = cleaner.clean_trace(trace)
                
            assert result["confidence"] == confidence
    
    def test_response_structure_validation(self, cleaner, mock_client):
        """Test that all responses have required fields."""
        test_trace = "Test trace"
        
        with patch.object(cleaner, 'client', mock_client):
            mock_client.clean_trace.return_value = {
                "cleaned_trace": "Cleaned version",
                "suspect_functions": ["func1", "func2"],
                "confidence": "High",
                "next_actions": ["Action 1", "Action 2"]
            }
            result = cleaner.clean_trace(test_trace)
            
        # Verify all required fields are present
        assert "cleaned_trace" in result
        assert "suspect_functions" in result
        assert "confidence" in result
        assert "next_actions" in result
        assert isinstance(result["suspect_functions"], list)
        assert isinstance(result["next_actions"], list)