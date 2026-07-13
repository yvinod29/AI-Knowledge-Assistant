import requests

from app.config import OLLAMA_BASE_URL, GEN_MODEL

PROMPT_TEMPLATE = """You are a document Q&A assistant. Answer the question using ONLY the context below.
If the context doesn't contain the answer, say you don't know — do not make anything up.
Cite the source page(s) you used in your answer, e.g. (Source: {{filename}}, p.{{page}}).

Context:
{context}

Question: {question}

Answer:"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        parts.append(
            f"[{chunk['source']}, p.{chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, chunks: list[dict]) -> str:
    context = build_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()
