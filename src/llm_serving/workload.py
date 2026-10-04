"""Workload generators.

Prompts are sent to the server as **token-id lists**, not text, so the prompt
length is exact and independent of tokenizer round-trips.

Which workload to use depends on what is being measured:

* ``synthetic``     random token ids.  Content-independent, so right for KV
                    capacity / throughput (Topic 2).  Never shares prefixes.
                    NOT valid for speculative decoding: random text gives
                    near-zero acceptance for every method.
* ``corpus``        random windows of a real text file (e.g. WikiText).  Use
                    this for speculation sweeps (Topic 3).
* ``sharegpt``      real chat prompts with a natural length distribution.
* ``shared_prefix`` synthetic, but a fraction of every prompt is identical.
                    Only for the prefix-caching ablation.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

TokenizerFn = Callable[[str], list[int]]


@dataclass
class Request:
    prompt_token_ids: list[int]
    max_tokens: int


class Workload(Protocol):
    def make_requests(
        self, n: int, context_len: int, output_len: int, seed: int
    ) -> list[Request]: ...


class SyntheticWorkload:
    # Ids in [1000, 30000) are ordinary tokens in Llama-2/3, GPT-2 and most
    # other vocabularies, and stay clear of special tokens.
    def __init__(self, token_low: int = 1000, token_high: int = 30000):
        self.low, self.high = token_low, token_high

    def make_requests(self, n, context_len, output_len, seed):
        rng = np.random.default_rng(seed)
        ids = rng.integers(self.low, self.high, size=(n, context_len)).tolist()
        return [Request(p, output_len) for p in ids]


class SharedPrefixWorkload:
    """Synthetic prompts whose first ``fraction`` of tokens are identical."""

    def __init__(self, fraction: float = 0.5, token_low: int = 1000, token_high: int = 30000):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1]")
        self.fraction, self.low, self.high = fraction, token_low, token_high

    def make_requests(self, n, context_len, output_len, seed):
        rng = np.random.default_rng(seed)
        n_shared = int(context_len * self.fraction)
        prefix = rng.integers(self.low, self.high, size=n_shared).tolist()
        suffixes = rng.integers(self.low, self.high, size=(n, context_len - n_shared)).tolist()
        return [Request(prefix + s, output_len) for s in suffixes]


class CorpusWorkload:
    """Random fixed-length windows of a pre-tokenized text corpus."""

    def __init__(self, token_ids: list[int] | np.ndarray):
        self.tokens = np.asarray(token_ids, dtype=np.int64)

    def make_requests(self, n, context_len, output_len, seed):
        if len(self.tokens) <= context_len:
            raise ValueError(
                f"corpus has {len(self.tokens)} tokens, need more than context_len={context_len}"
            )
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, len(self.tokens) - context_len, size=n)
        return [
            Request(self.tokens[s : s + context_len].tolist(), output_len) for s in starts
        ]


class ShareGPTWorkload:
    """Chat prompts with a natural length distribution.

    ``context_len`` acts as a *cap*: longer prompts are truncated, shorter ones
    are used as-is.  So the context-length axis means "max prompt length".
    """

    def __init__(self, prompts: list[list[int]], min_len: int = 16):
        self.prompts = [p for p in prompts if len(p) >= min_len]
        if not self.prompts:
            raise ValueError("no ShareGPT prompts left after length filtering")

    def make_requests(self, n, context_len, output_len, seed):
        rng = random.Random(seed)
        return [
            Request(rng.choice(self.prompts)[:context_len], output_len) for _ in range(n)
        ]


# ---------------------------------------------------------------------------
# Loading helpers (need a tokenizer, so they are kept out of the classes)
# ---------------------------------------------------------------------------

def hf_tokenizer_fn(name: str) -> TokenizerFn:
    """Return ``text -> token ids`` using a HuggingFace tokenizer."""
    from transformers import AutoTokenizer  # imported lazily: heavy dependency

    tok = AutoTokenizer.from_pretrained(name)
    return lambda text: tok.encode(text, add_special_tokens=False)


def load_corpus_tokens(path: str | Path, tokenize: TokenizerFn) -> list[int]:
    text = Path(path).read_text(encoding="utf-8")
    return tokenize(text)


def load_sharegpt_prompts(path: str | Path, tokenize: TokenizerFn, limit: int = 5000):
    """First human turn of each conversation in a ShareGPT-format JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts: list[list[int]] = []
    for item in data:
        for turn in item.get("conversations", []):
            if turn.get("from") in ("human", "user"):
                prompts.append(tokenize(turn["value"]))
                break
        if len(prompts) >= limit:
            break
    return prompts


def build_workload(
    kind: str,
    data_paths: dict[str, str],
    args: dict | None = None,
    tokenize: TokenizerFn | None = None,
) -> Workload:
    args = args or {}
    if kind == "synthetic":
        return SyntheticWorkload()
    if kind == "shared_prefix":
        return SharedPrefixWorkload(fraction=float(args.get("shared_prefix_fraction", 0.5)))
    if tokenize is None:
        raise ValueError(f"workload '{kind}' needs a tokenizer function")
    if kind == "corpus":
        return CorpusWorkload(load_corpus_tokens(data_paths["corpus_path"], tokenize))
    if kind == "sharegpt":
        return ShareGPTWorkload(load_sharegpt_prompts(data_paths["sharegpt_path"], tokenize))
    raise ValueError(f"unknown workload '{kind}'")
