import fitz
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
import time

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_pages_from_pdf(file_path: str) -> list:
    try:
        doc = fitz.open(file_path)
        pages = []
        
        for page in doc:
            text = page.get_text().strip()
            if text:
                pages.append(text)
        
        doc.close()    
        return pages
    except Exception as e:
        print(f"Error extracting pages from PDF: {e}")
        return []
    
    
def embed_text(pages: list[str], doc_title: str = "Uploaded Document") -> list[dict]:
    try:
        full_text = "\n\n".join(pages)
        
        separators = [
            "\n\n\n", # Very large breaks (e.g., between major sections)
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence endings
            "? ",
            "! ",
            " ",     # Word breaks
            ""       # Character fallback
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=separators
        )
        
        docs = text_splitter.create_documents([full_text])
        
        all_chunks = [doc.page_content for doc in docs]
        
        # vectors = embedding_model.encode(all_chunks, convert_to_numpy=True)
        
        # return all_chunks
        
        return [
            {
                "id": str(uuid.uuid4()),
                # "values": vectors[i].tolist(),
                "chunk_text": all_chunks[i],
                "doc_title": doc_title,
                
            }
            for i in range(len(all_chunks))
        ]
        
    except Exception as e:
        print(f"Error embedding text: {e}")
        return []
    
def upsert_embeddings_to_db(embeddings: list):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = PINECONE_INDEX_NAME
    
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

    dense_index = pc.Index(index_name)
    
    try:
        dense_index.upsert_records("chatpdf", records=embeddings)
    
        time.sleep(10)
    
        stats = dense_index.describe_index_stats()
        print(f"Index stats after upsert: {stats}")
    except Exception as e:
        print(f"Error upserting embeddings to DB: {e}")
    
    # batch_size = 100
    # try:
    #     for i in range(0, len(embeddings), batch_size):
    #         batch = embeddings[i:i + batch_size]
    #         index.upsert(vectors=batch, namespace="chatpdf")
            
    # except Exception as e:
    #     print(f"Error upserting embeddings to DB: {e}")
    #     return []