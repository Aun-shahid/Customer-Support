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


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=0.1
)

@router.post("/save-pdf")
async def save_pdf():
    print("Processing PDF...")
    try:
        doc = fitz.open("witcher.pdf")
        
        tokenizer = tiktoken.get_encoding("cl100k_base")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=lambda x: len(tokenizer.encode(x)),
            separators=["\n\n", "\n", " ", ""]
        )
        
        all_documents = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            
            # Split the text from this page into chunks
            page_chunks = text_splitter.split_text(page_text)
            
            # Create LangChain Document objects with metadata
            for i, chunk in enumerate(page_chunks):
                all_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"page_number": page_num + 1, "source": "witcher.pdf", "chunk_index": i}
                    )
                )
        doc.close()

        # Upsert Documents to Pinecone
        # PineconeVectorStore.from_documents expects a list of LangChain Document objects
        vectorstore = await asyncio.to_thread(
            PineconeVectorStore.from_documents,
            documents=all_documents,
            embedding=embeddings,
            index_name=index_name,
        )

        return {"Message": f"PDF processed and saved, total chunks {len(all_documents)}"}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="witcher.pdf not found in the current directory.")
    except Exception as e:
        print(f"Error in save_pdf: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during PDF processing: {e}")


@router.post("/chat") 
async def chat_with_document(query: str):
    try:
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        template = """You are a helpful AI assistant specialized in witcher book.
        Use the following context to answer the question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}

        Helpful Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        response = await asyncio.to_thread(qa_chain.invoke, {"query": query})
        
        # Correctly extract page_content and page_number from metadata
        source_info = []
        for doc in response["source_documents"]:
            source_info.append({
                "text": doc.page_content,
                "page_number": doc.metadata.get("page_number", "N/A"),
                "source": doc.metadata.get("source", "N/A")
            })
            
        return {
            "answer": response["result"],
            "source_documents": source_info
        }

    except Exception as e:
        print(f"Error in chat_with_document: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during chat: {e}")
