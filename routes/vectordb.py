from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY


pc = Pinecone(api_key=PINECONE_API_KEY)



index_name = "customer-support-index"

if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )