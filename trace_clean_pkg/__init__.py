"""trace-clean: Post-Mortem Stack-Trace Summarizer"""

__version__ = "1.0.0"

from .cli import TraceClean, main

__all__ = ["TraceClean", "main"]