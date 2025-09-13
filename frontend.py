import streamlit as st
import requests
import time
from typing import List

# -----------------------
# Configuration
# -----------------------
API_BASE_URL = "http://127.0.0.1:8000"
UPLOAD_URL = f"{API_BASE_URL}/upload"
ASK_URL = f"{API_BASE_URL}/ask"
STATUS_URL = f"{API_BASE_URL}/status"
CLEAR_URL = f"{API_BASE_URL}/clear"

# -----------------------
# Helper Functions
# -----------------------
def check_backend_status():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_session_status():
    """Get current session status"""
    try:
        response = requests.get(STATUS_URL)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def upload_files(files):
    """Upload PDF files to backend"""
    files_data = []
    for file in files:
        files_data.append(("files", (file.name, file.getvalue(), "application/pdf")))
    
    try:
        response = requests.post(UPLOAD_URL, files=files_data)
        return response
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        return None

def clear_session():
    """Clear current session"""
    try:
        response = requests.post(CLEAR_URL)
        return response.status_code == 200
    except:
        return False

# -----------------------
# Streamlit Configuration
# -----------------------
st.set_page_config(
    page_title="Dynamic RAG Chatbot", 
    page_icon="📚", 
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #1f77b4;
    margin-bottom: 30px;
}
.upload-section {
    background-color: #f0f8ff;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.chat-section {
    background-color: #f9f9f9;
    padding: 20px;
    border-radius: 10px;
}
.status-box {
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
.status-success {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
}
.status-warning {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Main UI
# -----------------------
st.markdown('<h1 class="main-header">📚 Dynamic RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown("Upload your PDF documents and ask questions based on their content!")

# Check backend status
if not check_backend_status():
    st.error("❌ Backend server is not running. Please start the backend first with: `uvicorn main:app --reload`")
    st.stop()

# Get current session status
session_status = get_session_status()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'documents_uploaded' not in st.session_state:
    st.session_state.documents_uploaded = False

# Update session state based on backend status
if session_status:
    st.session_state.documents_uploaded = session_status.get('has_documents', False)

# -----------------------
# Sidebar - Session Info
# -----------------------
with st.sidebar:
    st.header("📋 Session Info")
    
    if session_status and session_status.get('has_documents'):
        st.markdown('<div class="status-box status-success">✅ Documents are loaded</div>', unsafe_allow_html=True)
        st.write(f"**Session ID:** `{session_status.get('session_id', 'N/A')}`")
        
        # Show uploaded files
        if session_status.get('files_info'):
            st.write("**Uploaded Files:**")
            for file_info in session_status['files_info']:
                file_size_mb = file_info['size'] / (1024 * 1024)
                st.write(f"• {file_info['name']} ({file_size_mb:.1f} MB)")
        
        # Clear session button
        if st.button("🗑️ Clear Session", type="secondary"):
            if clear_session():
                st.session_state.documents_uploaded = False
                st.session_state.messages = []
                st.success("Session cleared successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to clear session")
    else:
        st.markdown('<div class="status-box status-warning">⚠️ No documents uploaded</div>', unsafe_allow_html=True)

# -----------------------
# Main Content Area
# -----------------------
# Create two columns
col1, col2 = st.columns([1, 2])

with col1:
    # -----------------------
    # File Upload Section
    # -----------------------
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.header("📄 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files (max 3 files, 10MB each)",
        type=['pdf'],
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    if uploaded_files:
        # Validate number of files
        if len(uploaded_files) > 3:
            st.error("❌ Please upload maximum 3 files")
        else:
            # Show file details
            st.write("**Selected files:**")
            total_size = 0
            for file in uploaded_files:
                file_size = len(file.getvalue()) / (1024 * 1024)  # MB
                total_size += file_size
                st.write(f"• {file.name} ({file_size:.1f} MB)")
            
            st.write(f"**Total size:** {total_size:.1f} MB")
            
            # Upload button
            if st.button("🚀 Process Documents", type="primary", key="upload_btn"):
                with st.spinner("Processing documents... This may take a few moments."):
                    response = upload_files(uploaded_files)
                    
                    if response and response.status_code == 200:
                        data = response.json()
                        st.success("✅ Documents processed successfully!")
                        st.info(f"Created {data['total_chunks']} text chunks from {len(data['files_processed'])} files")
                        st.session_state.documents_uploaded = True
                        time.sleep(2)
                        st.rerun()
                    else:
                        if response:
                            error_detail = response.json().get('detail', 'Unknown error')
                            st.error(f"❌ Upload failed: {error_detail}")
                        else:
                            st.error("❌ Failed to connect to backend")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # -----------------------
    # Chat Section
    # -----------------------
    st.markdown('<div class="chat-section">', unsafe_allow_html=True)
    st.header("💬 Ask Questions")
    
    if not st.session_state.documents_uploaded:
        st.warning("⚠️ Please upload PDF documents first to start asking questions.")
    else:
        # Chat interface
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show context if available
                if message["role"] == "assistant" and "context" in message:
                    with st.expander("📑 Show Retrieved Context"):
                        for i, chunk in enumerate(message["context"], 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(chunk["content"])
                            if chunk.get("source"):
                                st.caption(f"Source: {chunk['source']}")
                            st.divider()
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        payload = {
                            "question": prompt,
                            "include_context": True
                        }
                        response = requests.post(ASK_URL, json=payload)
                        
                        if response.status_code == 200:
                            data = response.json()
                            answer = data["answer"]
                            
                            st.markdown(answer)
                            
                            # Add assistant message to chat history
                            assistant_message = {
                                "role": "assistant", 
                                "content": answer
                            }
                            
                            # Add context if available
                            if data.get("retrieved"):
                                assistant_message["context"] = data["retrieved"]
                            
                            st.session_state.messages.append(assistant_message)
                            
                        else:
                            error_msg = f"Error {response.status_code}: {response.text}"
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"Sorry, I encountered an error: {error_msg}"
                            })
                    
                    except requests.exceptions.RequestException as e:
                        error_msg = f"Connection error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"Sorry, I couldn't connect to the backend: {error_msg}"
                        })
        
        # Clear chat button
        if st.session_state.messages:
            if st.button("🧹 Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("**Instructions:**")
st.markdown("1. Upload 1-3 PDF files (max 10MB each)")
st.markdown("2. Wait for processing to complete") 
st.markdown("3. Start asking questions about your documents!")
st.markdown("4. Use 'Clear Session' to upload new documents")