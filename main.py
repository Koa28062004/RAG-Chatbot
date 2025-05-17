import os
import google.generativeai as genai
import chromadb
import re
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb import Documents, EmbeddingFunction, Embeddings
from pathlib import Path
from tqdm import tqdm

# Set your Gemini API Key
os.environ["GEMINI_API_KEY"] = "AIzaSyBdy0GGjeUvFKV8GB09kNFbzAVkZK5BQE4"  # Replace with your key

# Load PDF with LangChain's PyPDFLoader
def load_pdfs_from_folder(folder_path: str) -> List:
    all_docs = []
    pdf_files = list(Path(folder_path).rglob("*.pdf"))  # Convert to list so tqdm works

    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = pdf_file.name
        all_docs.extend(docs)

    return all_docs

# Split documents into smaller chunks using LangChain's splitter
def split_documents(docs: List):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=lambda x: len(re.sub(r'\s+', ' ', x))
    )
    split_docs = splitter.split_documents(docs)
    # Add unique doc_id metadata
    for i, d in enumerate(split_docs):
        d.metadata["doc_id"] = i
    return split_docs

# Define custom embedding function using Gemini
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, inputs: List[str]) -> List[List[float]]:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=gemini_api_key)
        model = "models/text-embedding-004"
        title = "Document chunk"

        response = genai.embed_content(
            model=model,
            content=inputs,          # inputs is a list of strings
            task_type="retrieval_document",
            title=title
        )
        # response["embedding"] is expected to be a list of embedding vectors
        return response["embedding"]

# Create ChromaDB collection
def create_chroma_db(documents: List, path: str, name: str, batch_size: int = 50):
    chroma_client = chromadb.PersistentClient(path=path)
    embedding_fn = GeminiEmbeddingFunction()
    db = chroma_client.create_collection(name=name, embedding_function=embedding_fn)

    # Prepare batches of texts, ids, and metadatas
    texts = [d.page_content for d in documents]
    ids = [str(i) for i in range(len(documents))]
    metadatas = [d.metadata for d in documents]

    for i in tqdm(range(0, len(texts), batch_size), desc="Adding to ChromaDB in batches"):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]

        # Here, embedding is handled internally by Chroma using embedding_function,
        # so you only need to add documents normally.
        # If you want to pass embeddings yourself, you'd need to precompute embeddings.
        db.add(documents=batch_texts, ids=batch_ids, metadatas=batch_metadatas)

    return db



# Load an existing Chroma collection
def load_chroma_collection(path, name):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    return db

# Query ChromaDB for relevant chunks
def get_relevant_passages_with_metadata(query, db, n_results=100):
    results = db.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas"])
    docs = results['documents'][0]   # list of matched text chunks
    metadatas = results['metadatas'][0]  # list of metadata dicts for each chunk
    return docs, metadatas

# Build prompt for Gemini generation
def make_rag_prompt(query, relevant_passages):
    joined = " ".join(relevant_passages).replace('"', '').replace("'", '').replace('\n', ' ')
    prompt = f"""Bạn là trợ lý ảo giúp trả lời câu hỏi dựa trên các đoạn văn bản sau. Hãy tìm tất cả thông tin liên quan đến câu hỏi và trả lời thật chi tiết. Nếu thông tin không có trong đoạn văn, hãy ghi rõ "Không tìm thấy thông tin cụ thể".

Câu hỏi: {query}
Đoạn văn:
{joined}

Trả lời chi tiết:
"""
    return prompt

# Generate answer using Gemini model
def generate_gemini_answer(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not set.")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return response.text

# Full RAG answer generation pipeline
def generate_answer_with_source(db, query):
    docs, metadatas = get_relevant_passages_with_metadata(query, db)

    # Print source file and chunk text
    for i, (doc_text, meta) in enumerate(zip(docs, metadatas)):
        source_file = meta.get("source_file", "Unknown file")
        print(f"\n[Result chunk {i}] from file: {source_file}\nText:\n{doc_text}\n{'-'*40}")

    prompt = make_rag_prompt(query, docs)
    answer = generate_gemini_answer(prompt)
        
    return answer, docs, metadatas


# === Run the pipeline ===
if __name__ == "__main__":
    # Load & process PDF
    # pdf_path = "data"  # Replace with your PDF file
    # raw_docs = load_pdfs_from_folder(pdf_path)
    # split_docs = split_documents(raw_docs)

    # Create and save ChromaDB
    db_path = "db"  # folder to persist Chroma
    collection_name = "rag_experiment"
    # db = create_chroma_db(split_docs, path=db_path, name=collection_name)

    # Or later: load an existing collection
    db = load_chroma_collection(path=db_path, name=collection_name)

    # Ask your question
    question = "Những đối tượng áp dụng cho quy định của luật phòng cháy chữa cháy ?"
    answer, docs, metadatas = generate_answer_with_source(db, query=question)
    print("\n>> Gemini Answer:\n", answer)