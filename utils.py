import fitz
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import pc, PINECONE_INDEX_NAME

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
    try:
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
        dense_index.upsert_records("chatpdf", records=embeddings)
    
        # time.sleep(10)
    
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