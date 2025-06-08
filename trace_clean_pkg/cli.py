#!/usr/bin/env python3
"""
trace-clean: Post-Mortem Stack-Trace Summarizer
A lightweight CLI tool that provides concise summaries of raw stack traces.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import openai
import requests
import yaml
from rich.console import Console
from rich.json import JSON as RichJSON
from rich.markdown import Markdown

console = Console()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class TraceClean:
    """Main class for the trace-clean utility."""

    def __init__(
        self, model: str = "gpt-4o-mini", api_key: Optional[str] = None, local_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # Track if local_url was explicitly provided (not default)
        self._local_url_explicit = local_url != "http://localhost:11434"

        # Validate local URL
        if local_url:
            try:
                parsed = urlparse(local_url)
                if not parsed.scheme or not parsed.netloc:
                    raise ValueError(f"Invalid URL format: {local_url}")
            except Exception:
                raise ValueError(f"Invalid URL format: {local_url}") from None

        self.local_url = local_url
        self.config = self._load_config()

        if model != "local" and not self.api_key:
            raise ValueError(
                "API key required for non-local models. Set OPENAI_API_KEY environment variable or use --api-key flag."
            )

    def _extract_json_from_response(self, raw_response: str) -> Dict[str, Any]:
        """Extract JSON object from potentially messy model response."""
        # First try parsing the whole response (most common case)
        try:
            return json.loads(raw_response.strip())
        except json.JSONDecodeError:
            pass

        # Then try to find a complete JSON object
        stack = []
        start_idx = -1
        end_idx = -1

        for i, char in enumerate(raw_response):
            if char == "{":
                if not stack:
                    start_idx = i
                stack.append(char)
            elif char == "}":
                if stack:
                    stack.pop()
                    if not stack:
                        end_idx = i + 1
                        break

        if start_idx != -1 and end_idx != -1:
            try:
                json_str = raw_response[start_idx:end_idx]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # If that also fails, raise an error
        raise json.JSONDecodeError("Could not extract valid JSON from response", raw_response, 0)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from ~/.trace-clean/config.yaml if it exists."""
        config_path = Path.home() / ".trace-clean" / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}
                    # Apply config settings only if not explicitly provided in constructor
                    if "local_url" in config and not self._local_url_explicit:
                        self.local_url = config["local_url"]
                    return config
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        return {}

    def _get_few_shot_prompt(self, stack_trace: str) -> str:
        """Generate the few-shot prompt for the LLM."""
        prompt = (
            """You are an expert debugging assistant. Analyze the following stack trace and provide a structured summary.

Your response must be a valid JSON object with exactly these three fields:
1. "summary": A brief, plain-English explanation of the error (1-2 sentences)
2. "suspect_functions": An array of objects, each with:
   - "function": function name
   - "file": file path
   - "line": line number (integer)
   - "confidence": "High", "Medium", or "Low"
   - "reason": explanation of why this line is suspect
3. "next_actions": An array of 3-4 suggested debugging actions (strings)

Example 1:
Stack trace:
```
Traceback (most recent call last):
  File "app.py", line 25, in <module>
    main()
  File "app.py", line 21, in main
    result = process_data(data)
  File "app.py", line 15, in process_data
    return 100 / data['value']
KeyError: 'value'
```

Response:
{
  "summary": "The program crashed with a KeyError because it tried to access a dictionary key 'value' that doesn't exist in the process_data function.",
  "suspect_functions": [
    {
      "function": "process_data",
      "file": "app.py",
      "line": 15,
      "confidence": "High",
      "reason": "This line directly accesses data['value'] without checking if the key exists, causing the KeyError"
    },
    {
      "function": "main",
      "file": "app.py",
      "line": 21,
      "confidence": "Medium",
      "reason": "This function passes the data dictionary to process_data - the data might be missing the required 'value' key"
    }
  ],
  "next_actions": [
    "Add logging before line 15 to inspect the contents of the data dictionary",
    "Add a check for the 'value' key existence before accessing it",
    "Review where the data dictionary is created to ensure 'value' is included",
    "Set a breakpoint at line 15 to examine the data variable at runtime"
  ]
}

Example 2:
Stack trace:
```
Exception in thread "main" java.lang.NullPointerException
    at com.example.app.PricingService.calculateDiscount(PricingService.java:42)
    at com.example.app.CheckoutController.applyCustomerDiscount(CheckoutController.java:115)
    at com.example.app.CheckoutController.processCheckout(CheckoutController.java:87)
    at com.example.app.Main.main(Main.java:15)
```

Response:
{
  "summary": "A NullPointerException occurred in the calculateDiscount method, likely because a null object was accessed when trying to calculate a customer discount.",
  "suspect_functions": [
    {
      "function": "calculateDiscount",
      "file": "com/example/app/PricingService.java",
      "line": 42,
      "confidence": "High",
      "reason": "This is where the NullPointerException was thrown - likely accessing a method or property on a null object"
    },
    {
      "function": "applyCustomerDiscount",
      "file": "com/example/app/CheckoutController.java",
      "line": 115,
      "confidence": "Medium",
      "reason": "This method calls calculateDiscount and may be passing a null customer or pricing object"
    }
  ],
  "next_actions": [
    "Add null checks before line 42 in PricingService.java",
    "Add logging at the start of calculateDiscount to verify input parameters",
    "Review the customer object initialization in CheckoutController",
    "Run unit tests for PricingService with null inputs to reproduce the issue"
  ]
}

Now analyze the following stack trace:
```
"""
            + stack_trace
            + """
```

IMPORTANT: You must return ONLY valid JSON, nothing else. No text before or after. Start with { and end with }."""
        )
        return prompt

    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API."""
        client = openai.OpenAI(api_key=self.api_key)

        try:
            response = client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a debugging assistant that analyzes stack traces. Always respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.get("temperature", 0.3),
                max_tokens=self.config.get("max_tokens", 1500),
            )

            content = response.choices[0].message.content
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError("The model returned invalid JSON. Please try again.") from e
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _call_local_model(self, prompt: str) -> Dict[str, Any]:
        """Call local model via Ollama API."""
        url = f"{self.local_url}/api/generate"

        payload = {
            "model": self.config.get("local_model", "llama3.2"),
            "prompt": prompt,
            "stream": False,
            "temperature": self.config.get("temperature", 0.3),
            "options": {"num_predict": self.config.get("max_tokens", 2000)},
        }

        try:
            response = requests.post(url, json=payload, timeout=self.config.get("timeout", 120))
            response.raise_for_status()

            result = response.json()
            raw_response = result.get("response", "")

            # Try to extract JSON from the response
            return self._extract_json_from_response(raw_response)

        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to local model at {self.local_url}. Is Ollama running?") from None
        except json.JSONDecodeError:
            # Handle case where response.json() fails or JSON extraction fails
            if "response" in locals():
                try:
                    response_text = str(getattr(response, "text", "Unknown response"))[:500]
                except Exception:
                    response_text = "Unknown response"
            else:
                response_text = "Unknown response"
            logger.error(f"Failed to parse JSON from response: {response_text}...")
            raise ValueError("The local model returned invalid JSON. Please try again.") from None
        except Exception as e:
            logger.error(f"Local model error: {e}")
            raise

    def analyze(self, stack_trace: str) -> Dict[str, Any]:
        """Analyze a stack trace and return structured results."""
        prompt = self._get_few_shot_prompt(stack_trace)

        if self.model == "local":
            return self._call_local_model(prompt)
        else:
            return self._call_openai(prompt)

    def format_output(self, result: Dict[str, Any], json_format: bool = False) -> str:
        """Format the analysis result for display."""
        if json_format:
            return json.dumps(result, indent=2)

        # Create markdown output
        output = []
        output.append("# Stack Trace Analysis\n")

        # Summary
        output.append("## Summary")
        output.append(f"{result['summary']}\n")

        # Suspect Functions
        output.append("## Suspect Functions")
        for i, suspect in enumerate(result["suspect_functions"], 1):
            output.append(f"\n### {i}. {suspect['function']}")
            output.append(f"- **File**: `{suspect['file']}`")
            output.append(f"- **Line**: {suspect['line']}")
            output.append(f"- **Confidence**: {suspect['confidence']}")
            output.append(f"- **Reason**: {suspect['reason']}")

        # Next Actions
        output.append("\n## Next Actions")
        for i, action in enumerate(result["next_actions"], 1):
            output.append(f"{i}. {action}")

        return "\n".join(output)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="trace-clean: Post-Mortem Stack-Trace Summarizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  trace-clean error.log
  cat stacktrace.txt | trace-clean
  python app.py 2>&1 | trace-clean
  trace-clean --json < error.log
  trace-clean --model local --local-url http://localhost:11434
        """,
    )

    parser.add_argument("file", nargs="?", help="File containing the stack trace (reads from stdin if not provided)")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--model", default="gpt-4o-mini", help="Language model to use (default: gpt-4o-mini)")
    parser.add_argument("--api-key", help="API key for the language model provider")
    parser.add_argument(
        "--local-url",
        default="http://localhost:11434",
        help="URL for local model server (default: http://localhost:11434)",
    )

    args = parser.parse_args()

    # Read stack trace
    if args.file:
        if not os.path.exists(args.file):
            console.print(f"[red]Error: File '{args.file}' not found[/red]")
            sys.exit(1)
        with open(args.file, encoding="utf-8") as f:
            stack_trace = f.read()
    else:
        if sys.stdin.isatty():
            eof_key = "Ctrl+Z" if sys.platform == "win32" else "Ctrl+D"
            console.print(f"[yellow]Reading from stdin... (Press {eof_key} when done)[/yellow]")
        stack_trace = sys.stdin.read()

    if not stack_trace.strip():
        console.print("[red]Error: No stack trace provided[/red]")
        sys.exit(1)

    # Limit stack trace size to prevent memory issues
    MAX_STACK_TRACE_SIZE = 50000  # characters
    if len(stack_trace) > MAX_STACK_TRACE_SIZE:
        console.print(
            f"[yellow]Warning: Stack trace truncated from {len(stack_trace)} to {MAX_STACK_TRACE_SIZE} characters[/yellow]"
        )
        stack_trace = stack_trace[:MAX_STACK_TRACE_SIZE]

    try:
        # Create analyzer
        analyzer = TraceClean(model=args.model, api_key=args.api_key, local_url=args.local_url)

        # Analyze stack trace
        with console.status("[bold green]Analyzing stack trace..."):
            result = analyzer.analyze(stack_trace)

        # Format and display output
        if args.json:
            console.print(RichJSON(json.dumps(result, indent=2)))
        else:
            markdown = Markdown(analyzer.format_output(result))
            console.print(markdown)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
