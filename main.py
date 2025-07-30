from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import httpx
import fitz  # PyMuPDF
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List
import numpy as np
import faiss
import json

# ---- Load OpenAI Key ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
faiss_cache = {}

# ---- Auth Token ----
TEAM_TOKEN = "57d17fb3fd51f0068152497c4528563c4fefeee343ee094e4f2fe34b6b9cf096"

# ---- Data Models ----
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# ---- PDF Downloader ----
async def download_pdf_text(url: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
        response.raise_for_status()
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        doc = fitz.open("temp.pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        raise Exception(f"Download or extraction failed: {str(e)}")

# ---- Chunking ----
def split_text_into_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap if i + chunk_size < len(words) else len(words)
    return chunks

# ---- Embeddings ----
def get_embeddings(texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    embeddings = []
    for e in response.data:
        vec = np.array(e.embedding, dtype="float32")
        vec /= np.linalg.norm(vec)  # normalize vector
        embeddings.append(vec)
    return embeddings

# ---- FAISS Indexing ----
def build_faiss_index(chunks: List[str]):
    embeddings = get_embeddings(chunks)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, chunks

def search_faiss(index, query: str, chunks: List[str], k=10) -> List[str]:
    query_embedding = get_embeddings([query])[0]
    D, I = index.search(np.array([query_embedding]).astype("float32"), k)

    retrieved = [(chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]
    # Sort by distance (lower is better)
    reranked = sorted(retrieved, key=lambda x: x[1])
    return [chunk for chunk, _ in reranked[:5]]  # top 5 after re-ranking

# ---- Prompt Builder ----
def build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    return f"""
You are a legal assistant AI. Use the policy document context below to answer the question factually.

Context:
{context}

Question:
{question}

Respond in JSON format: {{"answer": "<your short, clear answer here>"}}
"""

# ---- LLM Call ----
def ask_llm(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always reply in JSON with an 'answer' key."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return json.dumps({"answer": f"Error generating answer: {str(e)}"})

# ---- API Endpoint ----
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or malformed.")

    token = authorization.split(" ")[1]
    if token != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    
    if request.documents in faiss_cache:
        index, chunk_list = faiss_cache[request.documents]
    else:
        document_text = await download_pdf_text(request.documents)
        chunks = split_text_into_chunks(document_text)
        index, chunk_list = build_faiss_index(chunks)
        faiss_cache[request.documents] = (index, chunk_list)

    try:
        answers = []
        for question in request.questions:
            relevant_chunks = search_faiss(index, question, chunk_list)
            prompt = build_prompt(question, relevant_chunks)
            llm_output = ask_llm(prompt)

            # ✅ Extract actual value from JSON response
            try:
                parsed = json.loads(llm_output)
                answers.append(parsed.get("answer", llm_output))
            except json.JSONDecodeError:
                answers.append(llm_output)

        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
