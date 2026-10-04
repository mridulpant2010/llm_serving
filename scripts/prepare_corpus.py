"""Download WikiText-2 and write the raw test/train text used by the harness.

    uv run python scripts/prepare_corpus.py

Needs internet and `pip install datasets`.  Writes data/wikitext2_{train,test}.txt.
"""

from pathlib import Path

from datasets import load_dataset

out = Path("data")
out.mkdir(exist_ok=True)
for split in ("train", "test"):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "".join(ds["text"])
    (out / f"wikitext2_{split}.txt").write_text(text, encoding="utf-8")
    print(f"{split}: {len(text):,} characters")
