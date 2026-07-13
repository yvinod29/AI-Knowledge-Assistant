"""
Matryoshka embedding dimension benchmark.

Inspired by the "Pocket Ranger" experiment (EmbeddingGemma, Samhitha Muvva):
tests whether truncating a Matryoshka-trained embedding (768 -> 512 -> 256 -> 128)
still retrieves the same documents as the full-size embedding, while using a
fraction of the storage.

Usage:
    python experiments/matryoshka_benchmark.py --pdf-dir data/uploads \
        --question "Do I need a reservation to enter in July?" \
        --question "What waterfalls are there?"

Requires the `embeddinggemma` Ollama model:
    ollama pull embeddinggemma
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embeddings import embed_text
from app.ingest import extract_chunks

MODEL = "embeddinggemma"
DIMENSIONS = [768, 512, 256, 128]
TOP_K = 3


def load_corpus(pdf_dir: str) -> list[dict]:
    chunks = []
    for path in glob.glob(os.path.join(pdf_dir, "*.pdf")):
        source = os.path.basename(path)
        chunks.extend(extract_chunks(path, source))
    return chunks


def truncate_and_normalize(vector: np.ndarray, dim: int) -> np.ndarray:
    truncated = vector[:dim]
    norm = np.linalg.norm(truncated)
    if norm > 0:
        truncated = truncated / norm
    return truncated


def build_dimension_matrices(full_embeddings: np.ndarray) -> dict[int, np.ndarray]:
    matrices = {}
    for dim in DIMENSIONS:
        rows = [truncate_and_normalize(vec, dim) for vec in full_embeddings]
        matrices[dim] = np.vstack(rows)
    return matrices


def top_k_indices(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    scores = matrix @ query_vec
    return list(np.argsort(-scores)[:k])


def run_benchmark(chunks: list[dict], questions: list[str]) -> None:
    print(f"Corpus: {len(chunks)} chunks from {len({c['source'] for c in chunks})} document(s)")
    print(f"Embedding model: {MODEL} (native 768 dims, Matryoshka-trained)\n")

    print("Embedding corpus at full 768 dimensions...")
    full_embeddings = np.array(
        [embed_text(chunk["text"], model=MODEL) for chunk in chunks], dtype=np.float32
    )
    matrices = build_dimension_matrices(full_embeddings)

    bytes_per_float = 4
    baseline_bytes = len(chunks) * DIMENSIONS[0] * bytes_per_float

    for question in questions:
        print(f"\n=== Question: {question!r} ===")
        query_full = np.array(embed_text(question, model=MODEL), dtype=np.float32)

        baseline_top_k = None
        print(f"{'dims':>6} | {'storage':>10} | {'vs 768':>7} | {'overlap':>8} | {'time (ms)':>10} | top chunks")
        for dim in DIMENSIONS:
            query_vec = truncate_and_normalize(query_full, dim)

            start = time.perf_counter()
            top_k = top_k_indices(query_vec, matrices[dim], TOP_K)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if dim == DIMENSIONS[0]:
                baseline_top_k = top_k
                overlap_pct = 100.0
            else:
                overlap = len(set(top_k) & set(baseline_top_k))
                overlap_pct = overlap / len(baseline_top_k) * 100

            storage_bytes = len(chunks) * dim * bytes_per_float
            storage_ratio = storage_bytes / baseline_bytes

            preview = ", ".join(
                f"{chunks[i]['source']} p.{chunks[i]['page']}" for i in top_k
            )
            print(
                f"{dim:>6} | {storage_bytes/1024:>7.1f} KB | {storage_ratio:>6.1%} | "
                f"{overlap_pct:>7.0f}% | {elapsed_ms:>10.3f} | {preview}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", default="data/uploads", help="Directory of PDFs to index")
    parser.add_argument(
        "--question", action="append", dest="questions", required=True,
        help="Question to test (repeat --question for multiple)",
    )
    args = parser.parse_args()

    chunks = load_corpus(args.pdf_dir)
    if not chunks:
        print(f"No PDF chunks found in {args.pdf_dir}. Upload a PDF via the app first.")
        return

    run_benchmark(chunks, args.questions)


if __name__ == "__main__":
    main()
