from typing import List
import os
import json
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load markdown files for text content
def load_text_documents(pdf_folder: str) -> List[dict]:
    text_docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    
    for fname in os.listdir(pdf_folder):
        if fname.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, fname)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()  # returns list of Langchain Document objects
            
            # documents usually have .page_content and .metadata
            for i, doc in enumerate(documents):
                chunks = splitter.split_text(doc.page_content)
                for j, chunk in enumerate(chunks):
                    text_docs.append({
                        "id": f"{fname.replace('.pdf', '')}_{i}_{j}",
                        "text": chunk,
                        "metadata": {
                            "type": "text",
                            "filename": fname,
                            "page": i,
                            "chunk": j
                        }
                    })
    return text_docs

# Load image metadata JSON
def load_json_documents(json_path: str, doc_type: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [{
        "id": entry["id"],
        "text": entry.get("content", "No Caption"),
        "metadata": {
            "type": doc_type,
            "url": entry.get("url", ""),
        }
    } for entry in raw]
