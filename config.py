import dotenv
import os
from pinecone import Pinecone

dotenv.load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = PINECONE_INDEX_NAME
    
if not pc.has_index(PINECONE_INDEX_NAME):
    pc.create_index_for_model(
        name=PINECONE_INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )

dense_index = pc.Index(PINECONE_INDEX_NAME)
