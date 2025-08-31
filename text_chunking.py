import re
import os
import datetime as dt
from typing import Iterable, List, Dict, Tuple

import pandas as pd

from utils_text import normalize_whitespace, extract_keyphrases


_SENT_SPLIT_RE = re.compile(
    r"(?<=[\.!?;:])\s+|\n{1,}|\u2022|\u2023|\u25CF|\u25E6|\u2219|\u2043|\-\s+"
)


def split_sentences_heuristic(text: str) -> List[str]:
    if not text:
        return []
    # Remove common page markers
    t = re.sub(r"---\s*Page\s+\d+\s*---", "\n", text, flags=re.IGNORECASE)
    parts = [normalize_whitespace(p) for p in _SENT_SPLIT_RE.split(t)]
    return [p for p in parts if p]


def chunk_by_sentences(
    text: str, max_chars: int = 600, overlap_sents: int = 1
) -> List[str]:
    sents = split_sentences_heuristic(text)
    if not sents:
        return []
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for s in sents:
        s_len = len(s) + (1 if cur else 0)
        if cur and (cur_len + s_len) > max_chars:
            chunks.append(" ".join(cur))
            # start new with overlap sentences
            if overlap_sents > 0:
                carry = cur[-overlap_sents:]
            else:
                carry = []
            cur = carry[:] if carry else []
            cur_len = sum(len(x) + 1 for x in cur) if cur else 0
        cur.append(s)
        cur_len += s_len
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def phrase_chunks(
    text: str,
    min_words: int = 2,
    max_words: int = 5,
    max_phrases: int = 50,
) -> List[str]:
    return extract_keyphrases(
        text, max_phrases=max_phrases, min_words=min_words, max_words=max_words
    )


def chunk_dataframe_sentences(
    df: pd.DataFrame,
    text_col: str,
    id_col: str,
    max_chars: int = 600,
    overlap_sents: int = 1,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, row in df.iterrows():
        doc_id = row.get(id_col)
        text = row.get(text_col) or ""
        for i, ch in enumerate(chunk_by_sentences(text, max_chars=max_chars, overlap_sents=overlap_sents)):
            rows.append(
                {
                    "source_id": doc_id,
                    "chunk_id": i,
                    "chunk_text": ch,
                    "chunk_type": "sentence",
                }
            )
    return pd.DataFrame(rows)


def chunk_dataframe_phrases(
    df: pd.DataFrame,
    text_col: str,
    id_col: str,
    min_words: int = 2,
    max_words: int = 5,
    max_phrases: int = 50,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, row in df.iterrows():
        doc_id = row.get(id_col)
        text = row.get(text_col) or ""
        phrases = phrase_chunks(
            text, min_words=min_words, max_words=max_words, max_phrases=max_phrases
        )
        for i, p in enumerate(phrases):
            rows.append(
                {
                    "source_id": doc_id,
                    "chunk_id": i,
                    "chunk_text": p,
                    "chunk_type": "phrase",
                }
            )
    return pd.DataFrame(rows)


def save_chunks_csv(df: pd.DataFrame, out_dir: str, prefix: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{prefix}_{ts}.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Chunk texts into sentence or 2-5 word phrase chunks")
    ap.add_argument("--input", required=True, help="Input CSV with text column")
    ap.add_argument("--text-col", default="extracted_text", help="Text column name")
    ap.add_argument("--id-col", default="file_name", help="ID column to carry over")
    ap.add_argument("--mode", choices=["sentences", "phrases", "both"], default="both")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--overlap-sents", type=int, default=1)
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--max-words", type=int, default=5)
    ap.add_argument("--max-phrases", type=int, default=50)
    args = ap.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8")
    out_paths: List[str] = []
    if args.mode in ("sentences", "both"):
        s_df = chunk_dataframe_sentences(
            df,
            text_col=args.text_col,
            id_col=args.id_col,
            max_chars=args.max_chars,
            overlap_sents=args.overlap_sents,
        )
        out_paths.append(save_chunks_csv(s_df, args.out_dir, "sentence_chunks"))

    if args.mode in ("phrases", "both"):
        p_df = chunk_dataframe_phrases(
            df,
            text_col=args.text_col,
            id_col=args.id_col,
            min_words=args.min_words,
            max_words=args.max_words,
            max_phrases=args.max_phrases,
        )
        out_paths.append(save_chunks_csv(p_df, args.out_dir, "phrase_chunks"))

    for p in out_paths:
        print(f"Wrote {len(pd.read_csv(p, encoding='utf-8'))} rows to {p}")

