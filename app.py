"""
Simple RAG System using Mistral:
PDF → Extract Text → FAISS → Query → JSON Response
"""

# -----------------------------
# 1) IMPORTS
# -----------------------------
import os
from dotenv import load_dotenv
import gradio as gr
from datetime import datetime

import fitz  # PyMuPDF for PDF text extraction

from mistralai import Mistral
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings


# -----------------------------
# 2) LOAD API & MISTRAL CLIENT
# -----------------------------
load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
if not API_KEY:
    raise ValueError("❌ Add MISTRAL_API_KEY in .env file.")

client = Mistral(api_key=API_KEY)

LLM_MODEL = "mistral-small"
EMBED_MODEL = "mistral-embed"

vector_db = None  # global store


# -----------------------------
# 3) PDF TEXT EXTRACTOR (NO OCR)
# -----------------------------
def extract_text_from_pdf(path: str) -> str:
    pdf = fitz.open(path)
    text = "\n".join(page.get_text() for page in pdf)
    
    if not text.strip():
        raise ValueError("❌ PDF contains no readable text. (Scanned?)")

    return text


# -----------------------------
# 4) EMBEDDINGS + FAISS DB
# -----------------------------
def store_to_vector_db(text: str) -> int:
    global vector_db

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError("❌ No valid text chunks generated.")

    embeddings = MistralAIEmbeddings(model=EMBED_MODEL, api_key=API_KEY)
    vector_db = FAISS.from_texts(chunks, embeddings)

    return len(chunks)


# -----------------------------
# 5) RAG ANSWERING (PROVER JSON)
# -----------------------------
def rag_answer(query: str):
    if vector_db is None:
        return {"error": "❌ Upload a PDF and index it first."}

    docs = vector_db.similarity_search_with_score(query, k=3)
    context = "\n\n".join(d.page_content for d, _ in docs)

    prompt = f"""
Answer using ONLY the following extracted document text.

CONTEXT:
{context}

QUESTION: {query}

If the answer is missing, reply exactly:
"Not available in document."
"""

    response = client.chat.complete(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    answer_text = response.choices[0].message.content.strip()

    evidence = [
        {"chunk": idx, "text": doc.page_content, "score": float(score)} 
        for idx, (doc, score) in enumerate(docs)
    ]

    return {
        "query": query,
        "answer": answer_text,
        "evidence": evidence,
        "timestamp": datetime.utcnow().isoformat(),
        "models": {"llm": LLM_MODEL, "embedding": EMBED_MODEL},
    }


# -----------------------------
# 6) GRADIO UI (2 BLOCKS)
# -----------------------------
def process_pdf(file):
    try:
        text = extract_text_from_pdf(file.name)
        count = store_to_vector_db(text)
        return f"📌 PDF processed and stored.\n🧩 Chunks created: {count}"
    except Exception as e:
        return f"❌ Error: {e}"


with gr.Blocks(title="PDF → FAISS → RAG (Mistral)") as ui:

    gr.Markdown("## 📁 Step 1 — Upload PDF → Extract Text → Build Vector DB")

    pdf_input = gr.File(label="Upload PDF File")
    status_box = gr.Textbox(label="Status", interactive=False)

    gr.Button("Process PDF").click(process_pdf, inputs=pdf_input, outputs=status_box)

    gr.Markdown("---")

    gr.Markdown("## 🔍 Step 2 — Ask Question from PDF Content")

    query_input = gr.Textbox(label="Ask a question...")
    json_output = gr.JSON(label="Prover JSON Response")

    gr.Button("Run RAG Query").click(rag_answer, inputs=query_input, outputs=json_output)

ui.launch()
