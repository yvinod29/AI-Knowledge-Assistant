"""
Programmatic use-case example: no HTTP server needed.

Ingests a PDF straight into ChromaDB and asks it a question, using the same
functions the FastAPI app calls internally (app/ingest.py, app/store.py,
app/llm.py). Useful for scripting, notebooks, or batch-processing many PDFs
without going through the /upload and /ask endpoints.

Usage:
    python examples/quickstart.py path/to/document.pdf "Your question here"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingest import ingest_pdf
from app.qa import answer_question


def main() -> None:
    if len(sys.argv) != 3:
        print('Usage: python examples/quickstart.py path/to/document.pdf "your question"')
        sys.exit(1)

    pdf_path, question = sys.argv[1], sys.argv[2]
    source_name = os.path.basename(pdf_path)

    print(f"Ingesting {source_name}...")
    chunk_count = ingest_pdf(pdf_path, source_name)
    print(f"Indexed {chunk_count} chunks.\n")

    print(f"Question: {question}")
    result = answer_question(question)
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")


if __name__ == "__main__":
    main()
