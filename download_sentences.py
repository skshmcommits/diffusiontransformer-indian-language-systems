"""
download_sentences.py
----------------------
Downloads a large corpus of clean English sentences and writes them
to data/sentences.txt, ready for the graph generator pipeline.

Source: WikiText-103 (via Hugging Face datasets)

Usage:
    python download_sentences.py                        # default: 100K sentences
    python download_sentences.py --max_sentences 10000  # quick test batch
    python download_sentences.py --max_sentences 500000 # large training run

Requirements:
    pip install datasets
"""

import os
import re
import argparse
from typing import Iterator

# ── Output path ───────────────────────────────────────────────────────────────
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sentences.txt")


def iter_sentences(text: str) -> Iterator[str]:
    """Split a paragraph into individual sentences."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    for sent in parts:
        sent = sent.strip()
        if re.search(r'[{}<>\[\]|]', sent):   # skip wiki markup
            continue
        yield sent


def is_good_sentence(sent: str, min_words: int, max_words: int) -> bool:
    """Basic quality filter."""
    words = sent.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    if not sent.isascii():             # reject Japanese, Chinese, Arabic, etc.
        return False
    if not sent[0].isupper():
        return False
    if sent[-1] not in '.!?':
        return False
    if re.search(r'\d{4,}', sent):     # skip long number sequences
        return False
    return True


def download_and_extract(
    max_sentences: int,
    dataset_name: str,
    min_words: int = 5,
    max_words: int = 25,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Run: pip install datasets")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[Downloader] Loading dataset  : {dataset_name}")
    print(f"[Downloader] Target           : {max_sentences:,} sentences")
    print(f"[Downloader] Word length range: {min_words}–{max_words} words")
    print(f"[Downloader] Output file      : {OUTPUT_FILE}\n")

    ds = load_dataset(dataset_name, "wikitext-103-raw-v1", split="train", streaming=True)

    count   = 0
    seen    = set()
    skipped = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in ds:
            text = item.get("text", "").strip()

            if not text or text.startswith("="):   # skip headers
                continue

            for sent in iter_sentences(text):
                if not is_good_sentence(sent, min_words, max_words):
                    skipped += 1
                    continue
                if sent in seen:
                    continue

                seen.add(sent)
                f.write(sent + "\n")
                count += 1

                if count % 5000 == 0:
                    print(f"  {count:>7,} collected  (filtered: {skipped:,})")

                if count >= max_sentences:
                    break

            if count >= max_sentences:
                break

    print(f"\n[Downloader] Done!")
    print(f"  Collected : {count:,} sentences")
    print(f"  Filtered  : {skipped:,} sentences (too short / long / malformed)")
    print(f"  Saved to  : {OUTPUT_FILE}")
    print(f"\nNext step:")
    print(f"  python main.py generate_data --sentences_file data/sentences.txt --output_dir data/generated_graphs/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download sentence corpus for OKM training")
    parser.add_argument("--max_sentences", type=int, default=100_000,
                        help="Number of sentences to collect (default: 100,000)")
    parser.add_argument("--dataset",       type=str, default="wikitext",
                        help="Hugging Face dataset name (default: wikitext)")
    parser.add_argument("--min_words",     type=int, default=5,
                        help="Minimum words per sentence (default: 5)")
    parser.add_argument("--max_words",     type=int, default=25,
                        help="Maximum words per sentence (default: 25)")
    args = parser.parse_args()

    download_and_extract(
        max_sentences=args.max_sentences,
        dataset_name=args.dataset,
        min_words=args.min_words,
        max_words=args.max_words,
    )


if __name__ == "__main__":
    main()
