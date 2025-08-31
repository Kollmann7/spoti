import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from parse_rome_sql import iter_metier_front
from utils_text import normalize_text, safe_join


def _decode_json_field(raw: str):
    if raw is None:
        return None
    # Raw comes from SQL with backslashes preserved; try a permissive load
    try:
        # Replace escaped quotes and common sequences
        s = raw.encode("utf-8", errors="ignore").decode("unicode_escape")
        return json.loads(s)
    except Exception:
        try:
            return json.loads(raw)
        except Exception:
            return None


def _build_text_for_metier(rec: Dict) -> str:
    libelle = rec.get("libelle") or ""
    definition = rec.get("definition") or ""
    appellations_raw = rec.get("appellations")
    appellations = _decode_json_field(appellations_raw) if isinstance(appellations_raw, str) else None
    app_list = []
    if isinstance(appellations, list):
        for a in appellations:
            lab = a.get("libelle") if isinstance(a, dict) else None
            if lab:
                app_list.append(lab)
    text = safe_join([libelle, definition, "; ".join(app_list)], sep=". ")
    return normalize_text(text)


def _collect_competences(rec: Dict) -> List[Dict]:
    out: List[Dict] = []
    # 1) From explicit competencesMobilisees* columns when present
    for key in (
        "competencesMobilisees",
        "competencesMobiliseesPrincipales",
        "competencesMobiliseesEmergentes",
    ):
        raw = rec.get(key)
        if not raw or not isinstance(raw, str):
            continue
        try:
            data = _decode_json_field(raw)
        except Exception:
            data = None
        if not isinstance(data, list):
            continue
        for item in data:
            # The JSON seems to contain objects with keys like {'frequence': X, 'competence': {...}}
            comp = item.get("competence") if isinstance(item, dict) else None
            if not isinstance(comp, dict):
                continue
            code = comp.get("code") or comp.get("codeOgr")
            libelle = comp.get("libelle")
            ctype = comp.get("type")
            if not libelle or not code:
                continue
            out.append({"code": str(code), "libelle": libelle, "type": ctype})
    # 2) From nested competences in 'appellations' JSON (field: competencesCles)
    apps_raw = rec.get("appellations")
    if isinstance(apps_raw, str) and apps_raw:
        apps = _decode_json_field(apps_raw)
        if isinstance(apps, list):
            for a in apps:
                if not isinstance(a, dict):
                    continue
                ccles = a.get("competencesCles")
                if not isinstance(ccles, list):
                    continue
                for item in ccles:
                    comp = item.get("competence") if isinstance(item, dict) else None
                    if not isinstance(comp, dict):
                        continue
                    code = comp.get("code") or comp.get("codeOgr")
                    libelle = comp.get("libelle")
                    ctype = comp.get("type")
                    if not libelle or not code:
                        continue
                    out.append({"code": str(code), "libelle": libelle, "type": ctype})
    return out


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


@dataclass
class IndexPaths:
    base: str
    metiers_index: str
    metiers_meta: str
    competences_index: str
    competences_meta: str


def _make_paths(out_dir: str) -> IndexPaths:
    _ensure_dir(out_dir)
    return IndexPaths(
        base=out_dir,
        metiers_index=os.path.join(out_dir, "rome_metiers.faiss"),
        metiers_meta=os.path.join(out_dir, "rome_metiers_meta.jsonl"),
        competences_index=os.path.join(out_dir, "rome_competences.faiss"),
        competences_meta=os.path.join(out_dir, "rome_competences_meta.jsonl"),
    )


def _load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is required. Please install dependencies from requirements.txt"
        ) from e
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}'. Ensure it's available locally or network is allowed."
        ) from e


def _to_embeddings(model, texts: List[str], passage_prefix: str = "") -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype="float32")
    batch = [f"{passage_prefix}{t}" if passage_prefix else t for t in texts]
    emb = model.encode(
        batch,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    arr = np.asarray(emb)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype("float32")


def _save_faiss(index_path: str, vectors: np.ndarray):
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "faiss-cpu is required. Please install dependencies from requirements.txt"
        ) from e
    if vectors.size == 0 or vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("No vectors to index: got empty embeddings array.")
    d = vectors.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(vectors)
    faiss.write_index(idx, index_path)


def _write_jsonl(path: str, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(args: argparse.Namespace) -> None:
    paths = _make_paths(args.out_dir)
    # Parse SQL -> collect metiers and competences
    metiers_meta: List[Dict] = []
    comp_map: Dict[str, Dict] = {}
    for rec in iter_metier_front(args.sql):
        code = rec.get("code")
        libelle = rec.get("libelle")
        if not code or not libelle:
            continue
        metiers_meta.append(
            {
                "code": code,
                "libelle": libelle,
                "definition": rec.get("definition"),
                "appellations_raw": rec.get("appellations"),
            }
        )
        for c in _collect_competences(rec):
            comp_map[c["code"]] = c

    # Build texts for embeddings
    metier_texts = [_build_text_for_metier(m) for m in metiers_meta]
    comp_list = list(comp_map.values())
    comp_texts = [normalize_text(c["libelle"]) for c in comp_list]

    # Load model and embed
    model_name = args.model
    model = _load_model(model_name)
    # E5 works best with 'passage: ' prefix for indexed items
    passage_prefix = "passage: " if "e5" in model_name.lower() else ""

    print(f"Preparing embeddings: {len(metier_texts)} metiers, {len(comp_texts)} competences")
    metier_vecs = _to_embeddings(model, metier_texts, passage_prefix=passage_prefix)
    comp_vecs = _to_embeddings(model, comp_texts, passage_prefix=passage_prefix)

    # Save FAISS indexes and metadata
    _save_faiss(paths.metiers_index, metier_vecs)
    _write_jsonl(paths.metiers_meta, metiers_meta)

    if comp_vecs.size > 0 and comp_vecs.shape[0] > 0:
        _save_faiss(paths.competences_index, comp_vecs)
        _write_jsonl(paths.competences_meta, comp_list)
        print(
            f"Built {len(metiers_meta)} metiers and {len(comp_list)} competences indices in '{args.out_dir}'."
        )
    else:
        print(
            "Warning: No competences parsed/embedded. Built metiers index only. Check JSON decoding of competences." 
        )


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build FAISS indices for ROME metiers and competences from SQL dump")
    p.add_argument("--sql", required=True, help="Path to dump_rome.sql")
    p.add_argument("--out-dir", default="output/rome_index", help="Output directory for indices and metadata")
    p.add_argument(
        "--model",
        default="intfloat/multilingual-e5-base",
        help="Sentence-Transformers model name or local path",
    )
    return p


if __name__ == "__main__":
    args = make_argparser().parse_args()
    build(args)
