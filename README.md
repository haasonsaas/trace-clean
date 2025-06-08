# trace-clean: Post-Mortem Stack-Trace Summarizer

trace-clean is a lightweight command-line tool that transforms verbose stack traces into concise, actionable summaries. By leveraging large language models, it helps developers quickly understand errors, identify root causes, and determine next debugging steps.

## Features

- **Intelligent Summarization**: Converts cryptic stack traces into plain-English explanations
- **Root Cause Analysis**: Pinpoints the most likely source of errors with confidence levels
- **Actionable Suggestions**: Provides specific debugging commands and next steps
- **Multiple Output Formats**: Supports both formatted Markdown and JSON output
- **Offline Mode**: Works with local LLMs via Ollama for privacy-sensitive environments
- **Language Agnostic**: Handles stack traces from Python, Java, JavaScript, and more

## Installation

### Via pip

```bash
pip install trace-clean
```

### From source

```bash
git clone https://github.com/haasonsaas/trace-clean.git
cd trace-clean
pip install -e .
```

## Quick Start

### Basic Usage

```bash
# Analyze a stack trace from a file
trace-clean error.log

# Pipe output directly from a failing program
python buggy_app.py 2>&1 | trace-clean

# Read from stdin
trace-clean < stacktrace.txt
```

### Output Formats

```bash
# Default: Rich formatted Markdown
trace-clean error.log

# JSON output for programmatic use
trace-clean --json error.log
```

## Configuration

### API Key Setup

trace-clean requires an API key for OpenAI:

```bash
# Option 1: Environment variable (recommended)
export OPENAI_API_KEY="your-api-key"

# Option 2: Command line
trace-clean --api-key "your-api-key" error.log
```

### Offline Mode with Ollama

For offline usage or enhanced privacy:

1. Install and run [Ollama](https://ollama.ai)
2. Pull a suitable model:
   ```bash
   ollama pull llama3.2
   ```
3. Use trace-clean in local mode:
   ```bash
   trace-clean --model local error.log
   ```

### Configuration File

Create `~/.trace-clean/config.yaml` for persistent settings:

```yaml
# Local model configuration
local_model: llama3.2
local_url: http://localhost:11434

# Model parameters (applies to both OpenAI and local models)
temperature: 0.3
max_tokens: 1500  # For OpenAI, or num_predict for local models
timeout: 120      # Request timeout in seconds for local models
```

## Example Output

Given a Python KeyError stack trace, trace-clean provides:

### Summary
> The program crashed with a KeyError because it tried to access a dictionary key 'value' that doesn't exist in the process_data function.

### Suspect Functions

1. **process_data** (`app.py:15`)
   - Confidence: High
   - Reason: This line directly accesses data['value'] without checking if the key exists

2. **main** (`app.py:21`)
   - Confidence: Medium  
   - Reason: This function passes the data dictionary to process_data - the data might be missing the required 'value' key

### Next Actions
1. Add logging before line 15 to inspect the contents of the data dictionary
2. Add a check for the 'value' key existence before accessing it
3. Review where the data dictionary is created to ensure 'value' is included
4. Set a breakpoint at line 15 to examine the data variable at runtime

## Command Line Options

```
trace-clean [OPTIONS] [FILE]

Arguments:
  FILE                    File containing the stack trace (reads from stdin if not provided)

Options:
  --help                  Show this help message and exit
  --json                  Output results in JSON format
  --model MODEL           Language model to use (default: gpt-4o-mini)
  --api-key KEY          API key for the language model provider
  --local-url URL        URL for local model server (default: http://localhost:11434)
```

## Supported Stack Trace Formats

trace-clean handles various common stack trace formats:

- **Python**: Traceback format with file paths and line numbers
- **Java**: Exception stack traces with package paths
- **JavaScript/Node.js**: Error stacks with file locations
- **C/C++**: Segmentation faults and core dumps
- **Ruby**: Exception backtraces
- And more...

## Use Cases

- **Quick Debugging**: Get instant insights into production errors
- **Code Reviews**: Understand crash reports in pull requests
- **Team Collaboration**: Share structured error analyses
- **Learning Tool**: Understand unfamiliar error messages
- **CI/CD Integration**: Automated error analysis in pipelines

## Privacy and Security

⚠️ **Important**: Stack traces may contain sensitive information such as:
- File paths revealing system structure
- Variable values including passwords or API keys
- Database connection strings
- User data

**Recommendations**:
- **API Mode**: Stack traces are sent to OpenAI's API. Review and sanitize traces before analysis
- **Local Mode**: Use Ollama for sensitive traces - all processing stays on your machine
- **No Persistence**: trace-clean is stateless and doesn't store any data locally

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

MIT License - see LICENSE file for details