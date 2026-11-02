"""Download WikiText-2 and write the raw test/train text used by the harness.

    python scripts/prepare_corpus.py

Writes data/wikitext2_{train,test}.txt with full fallback support.
"""

from pathlib import Path
import urllib.request
import sys

out = Path("data")
out.mkdir(exist_ok=True)

# Direct raw fallback URLs (PyTorch official NLP dataset mirrors)
FALLBACK_URLS = {
    "test": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
    "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
}

for split in ("train", "test"):
    target_file = out / f"wikitext2_{split}.txt"
    if target_file.exists() and target_file.stat().st_size > 1000:
        print(f"{split}: already exists ({target_file.stat().st_size:,} bytes)")
        continue

    downloaded = False
    # Method 1: Hugging Face datasets with Salesforce/ namespace
    for ds_name in ("Salesforce/wikitext", "wikitext"):
        try:
            from datasets import load_dataset
            print(f"Fetching {split} via datasets.load_dataset('{ds_name}', 'wikitext-2-raw-v1')...")
            ds = load_dataset(ds_name, "wikitext-2-raw-v1", split=split)
            text = "".join(ds["text"])
            target_file.write_text(text, encoding="utf-8")
            print(f"{split}: successfully saved {len(text):,} characters")
            downloaded = True
            break
        except Exception as e:
            print(f"Notice: load_dataset('{ds_name}') failed: {e}")

    # Method 2: Direct HTTP download fallback
    if not downloaded:
        url = FALLBACK_URLS[split]
        print(f"Fallback: downloading directly from {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_text = resp.read().decode("utf-8")
                target_file.write_text(raw_text, encoding="utf-8")
                print(f"{split}: successfully downloaded {len(raw_text):,} characters")
                downloaded = True
        except Exception as e:
            print(f"ERROR: Failed direct download for {split}: {e}", file=sys.stderr)
            raise

print("WikiText-2 corpus preparation completed successfully.")
