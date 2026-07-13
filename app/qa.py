from app.config import TOP_K
from app.embeddings import embed_text
from app.llm import generate_answer
from app.store import query


def answer_question(question: str) -> dict:
    question_embedding = embed_text(question)
    chunks = query(question_embedding, TOP_K)

    if not chunks:
        return {"answer": "No documents have been indexed yet.", "sources": []}

    answer = generate_answer(question, chunks)
    sources = [{"source": c["source"], "page": c["page"]} for c in chunks]
    return {"answer": answer, "sources": sources}
