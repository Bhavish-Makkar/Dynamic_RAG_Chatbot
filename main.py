import os
import tempfile
import shutil
from typing import List, Optional
from datetime import datetime
import hashlib

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

# =========================
# App Initialization
# =========================
app = FastAPI(title="Dynamic RAG Chatbot", description="A FastAPI RAG chatbot with dynamic PDF upload", version="2.0")

# Enable CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
client = None
current_vectorstore = None
current_retriever = None
current_session_id = None
uploaded_files_info = []

# Load env vars
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in .env file")

# Configuration
MODEL_NAME = "openai/gpt-oss-20b"  # Using a more reliable model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Faster and reliable embedding model
MAX_FILES = 3
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file

# =========================
# Helper Functions
# =========================
def generate_session_id():
    """Generate a unique session ID based on timestamp"""
    return hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]

def process_pdf_files(uploaded_files: List[UploadFile]) -> List:
    """Process uploaded PDF files and return document chunks"""
    all_docs = []
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            # Save uploaded file temporarily
            temp_file_path = os.path.join(temp_dir, file.filename)
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Load PDF
            try:
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing {file.filename}: {str(e)}")
    
    if not all_docs:
        raise HTTPException(status_code=400, detail="No valid documents found in uploaded files")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
    )
    
    doc_chunks = text_splitter.split_documents(all_docs)
    return doc_chunks

def create_vectorstore(doc_chunks: List, session_id: str):
    """Create a new vectorstore from document chunks"""
    embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Create vectorstore in memory (faster for small datasets)
    vectorstore = Chroma.from_documents(
        doc_chunks,
        embedding_model,
        collection_name=f"session_{session_id}",
    )
    
    return vectorstore

# =========================
# Startup Event
# =========================
@app.on_event("startup")
def startup_event():
    """Initialize Groq client at startup."""
    global client
    client = Groq(api_key=GROQ_API_KEY)
    print("✅ Startup complete — Groq client ready.")

# =========================
# Request/Response Models
# =========================
class FileUploadResponse(BaseModel):
    message: str
    session_id: str
    files_processed: List[str]
    total_chunks: int

class AskRequest(BaseModel):
    question: str
    k: Optional[int] = 5
    include_context: Optional[bool] = False

class RetrievedChunk(BaseModel):
    content: str
    source: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    retrieved: Optional[List[RetrievedChunk]] = None

class SessionStatus(BaseModel):
    has_documents: bool
    session_id: Optional[str] = None
    files_info: List[dict] = []

# =========================
# Health Check Endpoint
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "message": "Dynamic RAG Backend is alive"}

# =========================
# File Upload Endpoint
# =========================
@app.post("/upload", response_model=FileUploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process PDF files to create a new vectorstore"""
    global current_vectorstore, current_retriever, current_session_id, uploaded_files_info
    
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} files allowed")
    
    # Validate files
    processed_files = []
    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
        
        # Reset file pointer to get accurate size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds size limit")
        
        processed_files.append({
            "name": file.filename,
            "size": file_size
        })
    
    try:
        # Process PDFs and create chunks
        doc_chunks = process_pdf_files(files)
        
        # Generate new session ID
        session_id = generate_session_id()
        
        # Create vectorstore
        vectorstore = create_vectorstore(doc_chunks, session_id)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        # Update global state
        current_vectorstore = vectorstore
        current_retriever = retriever
        current_session_id = session_id
        uploaded_files_info = processed_files
        
        return FileUploadResponse(
            message="Files processed successfully",
            session_id=session_id,
            files_processed=[f["name"] for f in processed_files],
            total_chunks=len(doc_chunks)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

# =========================
# Session Status Endpoint
# =========================
@app.get("/status", response_model=SessionStatus)
def get_status():
    """Get current session status"""
    return SessionStatus(
        has_documents=current_retriever is not None,
        session_id=current_session_id,
        files_info=uploaded_files_info
    )

# =========================
# Clear Session Endpoint
# =========================
@app.post("/clear")
def clear_session():
    """Clear current session and uploaded documents"""
    global current_vectorstore, current_retriever, current_session_id, uploaded_files_info
    
    current_vectorstore = None
    current_retriever = None
    current_session_id = None
    uploaded_files_info = []
    
    return {"message": "Session cleared successfully"}

# =========================
# Main /ask Endpoint
# =========================
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Handles Q&A requests using RAG pipeline."""
    global client, current_retriever
    
    if client is None:
        raise HTTPException(status_code=500, detail="Server not initialized yet")
    
    if current_retriever is None:
        raise HTTPException(status_code=400, detail="No documents uploaded. Please upload PDF files first.")
    
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question provided")
    
    # Retrieve documents
    try:
        relevant_docs = current_retriever.get_relevant_documents(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retriever error: {str(e)}")
    
    if not relevant_docs:
        return AskResponse(
            answer="I don't have enough information in the uploaded documents to answer your question.",
            retrieved=[] if req.include_context else None
        )
    
    # Prepare context
    context_list = []
    retrieved_chunks = []
    
    for doc in relevant_docs[:req.k or 5]:
        context_list.append(doc.page_content)
        if req.include_context:
            source = doc.metadata.get('source', 'Unknown')
            retrieved_chunks.append(RetrievedChunk(
                content=doc.page_content,
                source=os.path.basename(source) if source else None
            ))
    
    context_for_query = "\n\n".join(context_list)
    
    # Prompt
    system_prompt = """You are a helpful AI assistant that answers questions based on the provided context from uploaded documents.

Instructions:
- Only use the information from the provided context to answer questions
- If the answer is not in the context, say "I don't have enough information in the uploaded documents to answer this question"
- Be concise and accurate in your responses
- Do not make up information that isn't in the context"""

    user_prompt = f"""Context from uploaded documents:
{context_for_query}

Question: {question}

Please provide a clear and accurate answer based only on the information in the context above."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Query Groq API
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=1000
        )
        answer_text = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
    
    # Return result
    return AskResponse(
        answer=answer_text,
        retrieved=retrieved_chunks if req.include_context else None
    )