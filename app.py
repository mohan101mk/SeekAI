import pathlib
import streamlit as st
import uuid
import os
import tempfile
from backend import upsert_documents, query_documents, run_janitor

# --- 1. INITIALIZATION & CLEANUP ---
# Generate a unique Session ID for the browser tab
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.document_uploaded = False
    
    # Run the janitor silently once per new session to clean up yesterday's files
    run_janitor(max_age_hours=24)

st.title("SeekAI")
st.caption(f"Session ID: {st.session_state.session_id[:8]}...") # Just showing a snippet for debugging

# --- 2. SIDEBAR: FILE UPLOAD ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
    "Upload a Document", 
    type=["pdf", "txt", "docx", "csv", "md", "html", "json", "pptx"]
    )
    
    
    if uploaded_file and not st.session_state.document_uploaded:
        file_extension = pathlib.Path(uploaded_file.name).suffix
        with st.spinner("Encrypting and Indexing..."):
            # Streamlit uploads are stored in RAM. We need to save it to a temporary file 
            # so LlamaIndex (SimpleDirectoryReader) can read it from the hard drive.
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name
            
            # Send it to Qdrant using your backend function!
            upsert_documents(temp_path, st.session_state.session_id)
            
            # Delete the temporary file from the local server
            os.remove(temp_path)
            
            st.session_state.document_uploaded = True
            st.success("Document ready for questioning!")

# --- 3. CHAT INTERFACE ---
st.header("Ask Questions")

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What is this document about?"):
    if not st.session_state.document_uploaded:
        st.warning("Please upload a document first!")
    else:
        # 1. Show user's message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Get AI response using your backend function!
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_documents(prompt, st.session_state.session_id)
                st.markdown(str(response))
        
        # 3. Save AI response to history
        st.session_state.messages.append({"role": "assistant", "content": str(response)})
