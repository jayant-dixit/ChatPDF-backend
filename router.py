from fastapi import APIRouter, File, UploadFile
import os
from utils import embed_text, extract_pages_from_pdf, upsert_embeddings_to_db
from llm import generate_response

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    print(f"Received file: {file}")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    print(f"File saved to: {file_path}")
        
    pages = extract_pages_from_pdf(file_path)
    
    print(f"Extracted {len(pages)} pages from the PDF.")
    # print(f"First page content (truncated): {pages[0][:200]}")
    os.remove(file_path)
    
    chunks = embed_text(pages, doc_title=file.filename)
    
    print(f"Generated {len(chunks)} chunks for the document.")
    # print(f"First chunk metadata: {chunks}")
    
    upsert_embeddings_to_db(chunks)    
        
    return {"success": True, "filename": file.filename, "message": "File uploaded successfully"}


@router.get("/query")
async def request_data(query: str):
    try:
        print(f"Received query: {query}")
        results = generate_response(query)
        return {"success": True, "data": results}
    except Exception as e:
        print(f"Error handling request: {e}")
        return {"success": False, "error": str(e)}
    