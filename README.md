# 📚 Dynamic RAG Chatbot

A hands-on **Retrieval-Augmented Generation (RAG)** chatbot that allows you to **upload PDFs** and ask questions about their content in real-time.  
Built with **FastAPI**, **Streamlit**, **LangChain**, **ChromaDB**, and **Groq LLMs**.

---

## 🚀 Features
- **PDF Uploads** → Upload up to 3 PDF files (max 10MB each).
- **Dynamic RAG Pipeline**:
  - Smart **document chunking** (512 chars + 50 overlap)
  - **Embeddings** with `SentenceTransformers`
  - **Vector search** with ChromaDB
  - **Context-aware responses** powered by Groq LLM.
- **Interactive Chat** → Ask questions directly in the Streamlit UI.
- **Session Management** → Clear/reset sessions and documents anytime.
- **Context Transparency** → Option to view retrieved context chunks.

---

## 🏗️ Architecture
```
[PDF Uploads] → [Text Splitting] → [Embeddings] → [ChromaDB Vectorstore]
                                               ↘
                                           [Retriever] → [Groq LLM] → [Answer]
```

- **Backend:** FastAPI (`main.py`)
- **Frontend:** Streamlit (`frontend.py`)
- **Storage:** ChromaDB (in-memory)
- **LLM Provider:** Groq API

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/dynamic-rag-chatbot.git
cd dynamic-rag-chatbot
```

### 2️⃣ Create & activate virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set environment variables
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 5️⃣ Start the backend (FastAPI)
```bash
uvicorn main:app --reload
```

Backend will run at: `http://127.0.0.1:8000`

### 6️⃣ Start the frontend (Streamlit)
```bash
streamlit run frontend.py
```

Frontend will run at: `http://localhost:8501`

---

## 📖 Usage
1. Open the Streamlit app in your browser.
2. Upload **1–3 PDF files**.
3. Wait for processing (documents are chunked & embedded).
4. Start chatting by asking questions about your documents.
5. Expand **“Show Retrieved Context”** to verify grounded answers.

---

## 🛠️ Tech Stack
- **Backend:** FastAPI, LangChain, Groq API
- **Frontend:** Streamlit
- **Vector DB:** ChromaDB
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)

---







## 🙌 Acknowledgements
- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Groq](https://groq.com/)
- [Streamlit](https://streamlit.io/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

💡 Built with passion for **AI, RAG, and real-world problem solving** 🚀
