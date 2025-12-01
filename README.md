# clinical_rag
# 📄 AI Document Understanding System (OCR + RAG + FAISS)

This project extracts text from **PDFs and images (including scanned & handwritten)** using OCR, stores the text in a **FAISS Vector Database**, and enables question-answering using a **RAG (Retrieval-Augmented Generation)** pipeline.

The application includes a **Gradio-based UI** for uploading documents and querying extracted knowledge.  
Built using **Python, LangChain, FAISS, PyMuPDF, pytesseract, pdf2image, and Mistral/OpenAI API.**

---

## 🚀 Features

- 📥 Upload PDF or Image files (JPG, PNG, scanned PDFs)
- 🔍 OCR-based text extraction (supports handwriting using Tesseract)
- 🧩 Text chunking and embedding with FAISS vector database
- 🤖 Query using Retrieval-Augmented Generation (RAG)
- 🖥️ Simple UI powered by **Gradio**
- 💾 Persistent storage for embeddings

---

## 🛠 Tech Stack

| Component | Technology |
|----------|------------|
| OCR | Tesseract OCR + pytesseract |
| PDF Processing | PyMuPDF / pdf2image |
| Vector Store | FAISS |
| LLM Model | Mistral / OpenAI |
| RAG Framework | LangChain |
| UI | Gradio |

---

## 📦 Installation

### 1️⃣ Install Python Dependencies

```bash
pip install pytesseract pdf2image pymupdf mistralai gradio langchain langchain-community
