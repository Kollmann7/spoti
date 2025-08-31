import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from utils_text import extract_keyphrases, normalize_text


def _load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_index(index_path: str):
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "faiss-cpu is required. Please install dependencies from requirements.txt"
        ) from e
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)
    return faiss.read_index(index_path)


def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model


def _encode_queries(model, queries: List[str]) -> np.ndarray:
    qprefix = "query: " if "e5" in model.__class__.__name__.lower() or "e5" in str(model).lower() else "query: "
    # Force 'query: ' prefix for E5; harmless for others
    batch = [f"{qprefix}{normalize_text(q)}" for q in queries]
    embs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")


def search(
    model_name: str,
    index_path: str,
    meta_path: str,
    queries: List[str],
    top_k: int = 10,
    threshold: float = 0.3,
) -> List[List[Tuple[float, Dict]]]:
    idx = _load_index(index_path)
    meta = _load_jsonl(meta_path)
    model = _load_model(model_name)
    qvecs = _encode_queries(model, queries)
    D, I = idx.search(qvecs, top_k)
    results: List[List[Tuple[float, Dict]]] = []
    for di, ii in zip(D, I):
        cur: List[Tuple[float, Dict]] = []
        for score, j in zip(di, ii):
            if j < 0:
                continue
            if score < threshold:
                continue
            cur.append((float(score), meta[j]))
        results.append(cur)
    return results


def cmd_search(args: argparse.Namespace):
    idx_name = args.index
    base = args.index_dir
    if idx_name == "competences":
        index_path = os.path.join(base, "rome_competences.faiss")
        meta_path = os.path.join(base, "rome_competences_meta.jsonl")
    else:
        index_path = os.path.join(base, "rome_metiers.faiss")
        meta_path = os.path.join(base, "rome_metiers_meta.jsonl")

    res = search(args.model, index_path, meta_path, [args.query], top_k=args.top_k, threshold=args.threshold)[0]
    for score, item in res:
        code = item.get("code") or item.get("id")
        libelle = item.get("libelle")
        print(f"{score:.3f}\t{code}\t{libelle}")


def cmd_annotate(args: argparse.Namespace):
    import csv

    input_csv = args.input
    text_col = args.text_col
    id_col = args.id_col
    idx_name = args.index
    base = args.index_dir
    if idx_name == "competences":
        index_path = os.path.join(base, "rome_competences.faiss")
        meta_path = os.path.join(base, "rome_competences_meta.jsonl")
    else:
        index_path = os.path.join(base, "rome_metiers.faiss")
        meta_path = os.path.join(base, "rome_metiers_meta.jsonl")

    # Prepare index/model once
    idx = _load_index(index_path)
    meta = _load_jsonl(meta_path)
    model = _load_model(args.model)

    out_rows: List[Dict] = []
    with open(input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get(id_col) or ""
            text = row.get(text_col) or ""
            # Extract short phrases (2–5 words)
            phrases = extract_keyphrases(text, max_phrases=args.max_phrases, min_words=args.min_words, max_words=args.max_words)
            if not phrases:
                continue
            # Encode queries
            qvecs = _encode_queries(model, phrases)
            D, I = idx.search(qvecs, args.top_k)
            for p, di, ii in zip(phrases, D, I):
                for score, j in zip(di, ii):
                    if j < 0 or score < args.threshold:
                        continue
                    item = meta[j]
                    out_rows.append(
                        {
                            "source_id": rid,
                            "phrase": p,
                            "score": f"{float(score):.6f}",
                            "rome_code": item.get("code") or item.get("id"),
                            "rome_libelle": item.get("libelle"),
                            "index": idx_name,
                        }
                    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.output or f"matches_{idx_name}.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_id",
                "phrase",
                "score",
                "rome_code",
                "rome_libelle",
                "index",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} matches to {out_path}")


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Search and annotate against ROME FAISS indices")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_s = sub.add_parser("search", help="Search a single free-text query")
    p_s.add_argument("--index", choices=["metiers", "competences"], default="competences")
    p_s.add_argument("--index-dir", default="output/rome_index")
    p_s.add_argument("--model", default="intfloat/multilingual-e5-base")
    p_s.add_argument("--query", required=True)
    p_s.add_argument("--top-k", type=int, default=10)
    p_s.add_argument("--threshold", type=float, default=0.30)
    p_s.set_defaults(func=cmd_search)

    p_a = sub.add_parser("annotate", help="Batch annotate a CSV with best-matching ROME entries from phrases")
    p_a.add_argument("--index", choices=["metiers", "competences"], default="competences")
    p_a.add_argument("--index-dir", default="output/rome_index")
    p_a.add_argument("--model", default="intfloat/multilingual-e5-base")
    p_a.add_argument("--input", required=True, help="Input CSV (e.g., results/chunks_*.csv)")
    p_a.add_argument("--text-col", default="chunk_text", help="Text column to extract phrases from")
    p_a.add_argument("--id-col", default="doc_id", help="ID column to carry over")
    p_a.add_argument("--top-k", type=int, default=10)
    p_a.add_argument("--threshold", type=float, default=0.35)
    p_a.add_argument("--max-phrases", type=int, default=20)
    p_a.add_argument("--min-words", type=int, default=1)
    p_a.add_argument("--max-words", type=int, default=5)
    p_a.add_argument("--out-dir", default="results")
    p_a.add_argument("--output", default=None, help="Output filename (defaults to matches_<index>.csv)")
    p_a.set_defaults(func=cmd_annotate)

    return p


if __name__ == "__main__":
    args = make_argparser().parse_args()
    args.func(args)

