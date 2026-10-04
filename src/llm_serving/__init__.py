"""LLM Serving — KV Cache Precision & Speculative Decoding Experiments."""

__version__ = "0.1.0"


def main() -> None:
    """Console entry point (``llm-serving``); see ``llm_serving.cli``."""
    from .cli import main as _main

    _main()
