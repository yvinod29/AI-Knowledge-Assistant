import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import UPLOAD_DIR
from app.ingest import ingest_pdf
from app.qa import answer_question
from app.store import list_sources

app = FastAPI(title="AI Knowledge Assistant")

os.makedirs(UPLOAD_DIR, exist_ok=True)


class AskRequest(BaseModel):
    question: str


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest_path, "wb") as f:
        f.write(await file.read())

    chunk_count = ingest_pdf(dest_path, file.filename)
    return {"filename": file.filename, "chunks_indexed": chunk_count}


@app.post("/ask")
def ask(request: AskRequest):
    return answer_question(request.question)


@app.get("/sources")
def sources():
    return {"sources": list_sources()}


app.mount("/static", StaticFiles(directory="static"), name="static")
