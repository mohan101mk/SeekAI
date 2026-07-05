from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient

from llama_index.llms.groq import Groq
import time

from qdrant_client import models
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter

import os
from dotenv import load_dotenv

load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

def initialize_backend():
    """Initializes LLM and Qdrant client."""
    # LLM Setup
    Settings.llm = Groq(model="openai/gpt-oss-120b", api_key=os.getenv('groq'))
    
    # Qdrant Client Setup
    client = QdrantClient(
        url=os.getenv('qdrant_endpoint'), 
        api_key=os.getenv('qdrant_api')
    )
    
    return client

def upsert_documents(file_path, group_id):
    """
    Reads a document, tags it with a group_id, and upserts to Qdrant.
    """
    client = initialize_backend()
    
    # Setup Vector Store with the shared collection name
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name="user_data_cluster"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Read and split document
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    
    # Add group_id metadata to every chunk
    current_time=int(time.time())
    for doc in documents:
        doc.metadata["group_id"] = group_id
        doc.metadata["file_name"] = os.path.basename(file_path)
        doc.metadata["uploaded_at"] = current_time
        
    # Create and return index
    
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context
    )

    try:
        client.create_payload_index(
            collection_name="user_data_cluster",
            field_name="group_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name="user_data_cluster",
            field_name="uploaded_at",
            field_schema=models.PayloadSchemaType.INTEGER
        )
    except Exception:
        pass # If the index already exists, just ignore and move on

    return index



def query_documents(prompt, group_id):
    """
    Searches the Qdrant database using the specific group_id filter.
    """
    client = initialize_backend()

    # Connect to the existing vector store
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name="user_data_cluster"
    )

    # Load the index from the vector store
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Create the filter for this specific user
    query_filter = MetadataFilters(
    filters=[
        ExactMatchFilter(
            key="group_id",
            value=group_id
        )
    ]
    
  )
    points, _ = client.scroll(
    collection_name="user_data_cluster",
    scroll_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="group_id",
                match=models.MatchValue(value=group_id)
            )
        ]
    ),
    limit=1
  )

    if len(points) == 0:
      return "No document uploaded"

    # Define the custom System Prompt template
    qa_prompt_tmpl_str = (
        "You are a helpful, professional assistant analyzing documents.\n"
        "Here is the context from the user's document:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Using ONLY the context above, answer the following question.\n"
        "If you don't know the answer, just say 'I cannot find the answer in your document.'\n"
        "Question: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    retriever = index.as_retriever(
      filters=query_filter,
      similarity_top_k=3
    )

    nodes = retriever.retrieve(prompt)

    if len(nodes) == 0:
      return "I cannot find the answer in your document."

    # Build the query engine with the filter and custom prompt
    query_engine = index.as_query_engine(
        filters=query_filter,
        similarity_top_k=3, # Retrieves the top 3 most relevant chunks
        text_qa_template=qa_prompt_tmpl # Injecting the system prompt here
    )
    

    return query_engine.query(prompt)


def run_janitor(max_age_hours=24):
    """
    Deletes all document chunks from Qdrant that are older than max_age_hours.
    """
    client = initialize_backend()
    cutoff_time = int(time.time()) - (max_age_hours * 3600)
    
    try:
        client.delete(
            collection_name="user_data_cluster",
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="uploaded_at",
                            range=models.Range(lt=cutoff_time)
                        )
                    ]
                )
            )
        )

    except Exception as e:
        print(f"Janitor encountered an error: {e}")