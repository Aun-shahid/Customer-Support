from fastapi import APIRouter, Depends, HTTPException
from config import OPENAI_API_KEY, PINECONE_API_KEY
from pinecone import Pinecone, ServerlessSpec
import fitz
import asyncio

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Import Document class
import tiktoken

pc = Pinecone(api_key=PINECONE_API_KEY)

router = APIRouter()

index_name = "customer-support-index"

try:
    if index_name not in pc.list_indexes():
        print(f"Index '{index_name}' does not exist. Creating it now...")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists. Skipping creation.")
except Exception as e:
    if "409" in str(e) or "already exists" in str(e).lower():
        print(f"Index '{index_name}' already exists. (Caught 409 Conflict error, proceeding normally).")
    else:
        print(f"An unexpected error occurred during index creation: {e}")
        # Optionally re-raise if this is a critical error for app startup
        # raise

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=0.1
)


@router.post("/semantic_search_page") # Using POST for more flexibility
async def semantic_search_page(query: str, k: int = 5): # k = number of results
    """
    Performs a semantic search and returns relevant chunks with page numbers.
    """
    try:
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        
        # Perform similarity search directly
        # You can specify k (number of results) here
        results = await asyncio.to_thread(vectorstore.similarity_search, query, k=k)
        
        search_results = []
        for doc in results:
            search_results.append({
                "text": doc.page_content,
                "page_number": doc.metadata.get("page_number", "N/A"),
                "source": doc.metadata.get("source", "N/A")
            })
            
        return {"query": query, "results": search_results}

    except Exception as e:
        print(f"Error in semantic_search_page: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during semantic search: {e}")