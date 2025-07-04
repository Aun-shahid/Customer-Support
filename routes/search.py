from fastapi import APIRouter, Depends
from config import OPENAI_API_KEY, PINECONE_API_KEY
from pinecone import Pinecone, ServerlessSpec


pc = Pinecone(api_key=PINECONE_API_KEY)

router = APIRouter()


index_name = "customer-support-index"



@router.get("/initialize")
async def initialize_index():
    if not pc.has_index(index_name):
        pc.create_index(
            index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(),
        )


@router.post("/save/{text}")
async def search(text: str):

    

    if PINECONE_API_KEY:
        print("Pinecone API Key is set.")
        
    if OPENAI_API_KEY:
        print("OpenAI API Key is set.")
    return {"message": "Search endpoint"}