from dotenv import load_dotenv
import os
import tempfile
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import requests
from typing import Optional

load_dotenv()
app = FastAPI()

class RequestBody(BaseModel):
    documents: str
    questions: list[str]

def process_documents(file_url, chunk_size=1000):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = file_url.split("?")[0].split("/")[-1]
        file_path = os.path.join(temp_dir, file_name)

        response = requests.get(file_url)
        if response.status_code != 200:
            raise ValueError("Failed to download document")
        with open(file_path, "wb") as f:
            f.write(response.content)

        docs = []
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["exact_location"] = f"Page {doc.metadata.get('page', 'N/A')}"
            docs.extend(loaded_docs)
        elif file_path.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
            docs.extend(loader.load())
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
            docs.extend(loader.load())
        else:
            raise ValueError("Unsupported file type")

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=chunk_size,
            chunk_overlap=200,
        )
        splits = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        return FAISS.from_documents(splits, embeddings)
    
@app.post("/hackrx/run")
async def run_hackrx(request: RequestBody, authorization: Optional[str] = Header(None)):
    # Only enforce authentication in production
    if os.getenv("ENV", "dev") == "prod":
        if not authorization or authorization != f"Bearer {os.getenv('API_KEY')}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    vectorstore = process_documents(request.documents)

    prompt_template = """You are an insurance claim assistant.
    Use the following context to answer the question.
    - Quote exact phrases.
    - Cite source name and location.
    - Reply in JSON with Decision (approved/rejected), Amount(if applicable), Justification, and Confidence Score.

    {context}

    Question: {question}
    Answer:"""

    QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatOpenAI(temperature=0.3, max_tokens=500, model_name="gpt-4.1-nano-2025-04-14")

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

    answers = []
    for question in request.questions:
        result = qa_chain({"query": question})
        answers.append(result["result"])

    return {"answers": answers}