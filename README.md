# SeekAI 🔍

SeekAI is a Retrieval-Augmented Generation (RAG) web application that allows users to upload documents and converse with them. Built with Streamlit, LlamaIndex, Qdrant, and Groq, it provides high-speed, context-aware answers based strictly on the contents of the uploaded files. 

Privacy and clean storage are built-in: user sessions are isolated using UUIDs, and a janitor function automatically scrubs document chunks older than 24 hours from the vector database.

## Features

* **Multi-Format Support:** Upload various document types including `.pdf`, `.txt`, `.docx`, `.csv`, `.md`, `.html`, `.json`, and `.pptx`.
* **Smart Retrieval:** Uses HuggingFace embeddings (`BAAI/bge-small-en-v1.5`) and Qdrant vector database for highly accurate semantic search.
* **Scalable Multitenancy:** Implements Qdrant's payload-based filtering to securely store all embeddings in a single shared collection. This optimizes database resources while ensuring strict logical separation of data using group_id metadata.
* **Session Isolation:** Every browser tab generates a unique Session ID, ensuring your queries only search against your own uploaded documents.
* **Automatic Data Cleanup:** A built-in janitor script automatically deletes vector data older than 24 hours to save space and maintain privacy.

## Project Structure

* `app.py`: The frontend application built with Streamlit. Handles the chat interface, file uploads, and session management.
* `backend.py`: The LlamaIndex backend pipeline. Manages the LLM/Embedding initialization, document chunking, Qdrant upsertion, contextual querying, and vector cleanup.
* `requirements.txt`: The complete list of Python dependencies required to run the application.

## Prerequisites

Before running the application, ensure you have the following:

1.  **Python 3.12.13** installed on your machine.
2.  A **Groq API Key** (for the LLM).
3.  A **Qdrant Cloud** cluster URL and API key (for the vector database).

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mohan101mk/SeekAI.git](https://github.com/mohan101mk/SeekAI.git)
   cd SeekAI
   ```
2. **Create a virtual environment (recommended):**
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  ```
3. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

## Configuration

Create a .env file in the root directory of the project and add your API keys:

  1. **groq**=your_groq_api_key_here
  2. **qdrant_endpoint**=your_qdrant_cluster_url_here
  3. **qdrant_api**=your_qdrant_api_key_here

## Usage

1. **Start the application:**
  ```bash
   streamlit run app.py
  ```
2. **Interact with SeekAI:**
   
  * Open the URL provided in your terminal (usually http://localhost:8502).
  * Use the sidebar on the left to upload your document.
  * Wait for the "Document ready for questioning!" success message.
  * Ask questions in the main chat interface based on the document's content.

## How It Works
  1. Ingestion: When a file is uploaded, Streamlit temporarily saves it to the disk. SimpleDirectoryReader loads the document, and the text is chunked and embedded using the bge-small-en-v1.5 model.
  2. Storage: The embeddings are pushed to Qdrant into a collection named user_data_cluster. Every chunk is tagged with your unique group_id (session ID) and an uploaded_at timestamp.  
  3. Retrieval: When you ask a question, SeekAI converts your prompt into an embedding, filters the Qdrant database using your specific group_id, and retrieves the top 3 most relevant chunks.
  4. Generation: The context and your question are passed to the Groq LLM using a strict prompt template that prevents hallucinations by forcing the model to rely only on the provided document context.





