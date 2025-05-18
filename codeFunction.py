import os
import json
from typing import List
from tqdm import tqdm
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini embedding function
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, inputs: List[str]) -> List[List[float]]:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=GEMINI_API_KEY)
        model = "models/text-embedding-004"
        response = genai.embed_content(
            model=model,
            content=inputs,
            task_type="retrieval_document",
            title="Document chunk"
        )
        return response["embedding"]

# ChromaDB wrapper
class ChromaDB:
    @staticmethod
    def create_chroma_db(documents: List[dict], path: str, name: str, batch_size: int = 50):
        chroma_client = chromadb.PersistentClient(path=path)
        embedding_fn = GeminiEmbeddingFunction()

        try:
            chroma_client.delete_collection(name)
        except:
            pass

        db = chroma_client.create_collection(name=name, embedding_function=embedding_fn)

        for i in tqdm(range(0, len(documents), batch_size), desc=f"Creating collection: {name}"):
            batch = documents[i:i + batch_size]
            db.add(
                documents=[item["text"] for item in batch],
                metadatas=[item["metadata"] for item in batch],
                ids=[item["id"] for item in batch]
            )
        return db

    @staticmethod
    def load_chroma_collection(path: str, name: str):
        chroma_client = chromadb.PersistentClient(path=path)
        return chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

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

def make_rag_prompt(query, context: str):
    prompt = f"""Bạn là trợ lý ảo giúp trả lời câu hỏi dựa trên các đoạn văn bản sau. Hãy tìm tất cả thông tin liên quan đến câu hỏi và trả lời thật chi tiết. Nếu thông tin không có trong đoạn văn, hãy ghi rõ "Không tìm thấy thông tin cụ thể".

Câu hỏi: {query}
Đoạn văn:
{context}

Trả lời chi tiết:
"""
    return prompt


def generate_gemini_answer(prompt: str):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return response.text


def generate_answer_with_source(text_db, image_db, table_db, query, text_n_results=100, image_n_results=3, table_n_results=3):
    text_res = text_db.query(query_texts=[query], n_results=text_n_results, include=["documents", "metadatas", "distances"])
    image_res = image_db.query(query_texts=[query], n_results=image_n_results, include=["documents", "metadatas", "distances"])
    table_res = table_db.query(query_texts=[query], n_results=table_n_results, include=["documents", "metadatas", "distances"])

    text_doc = text_res['documents'][0]
    image_doc = image_res['documents'][0]
    table_doc = table_res['documents'][0]

    text_metadata = text_res['metadatas'][0]
    image_metadata = image_res['metadatas'][0]
    table_metadata = table_res['metadatas'][0]

    text_distances = text_res['distances'][0]
    image_distances = image_res['distances'][0]
    table_distances = table_res['distances'][0]

    prompt = make_rag_prompt(query, text_doc)
    answer = generate_gemini_answer(prompt)

    images_res = []
    tables_res = []

    for i, (doc_text, meta, text_dis) in enumerate(zip(text_doc, text_metadata, text_distances)):
        source_file = meta.get("filename", "Unknown file")
        print(f"\n[Text chunk {i}] from file: {source_file} - Distance: {text_dis}\nText:\n{doc_text}\n{'-'*40}")

    for i, (doc_text, meta, image_dis) in enumerate(zip(image_doc, image_metadata, image_distances)):
        source_file = meta.get("url", "Unknown file")
        if image_dis < 0.1:
            images_res.append(source_file)
        print(f"\n[Image chunk {i}] from file: {source_file} - Distance: {image_dis}\nText:\n{doc_text}\n{'-'*40}")

    for i, (doc_text, meta, table_dis) in enumerate(zip(table_doc, table_metadata, table_distances)):
        source_file = meta.get("url", "Unknown file")
        if table_dis < 0.1:
            tables_res.append(source_file)
        print(f"\n[Table chunk {i}] from file: {source_file} - Distance: {table_dis}\nText:\n{doc_text}\n{'-'*40}")

    answer += "\n\n"
    if images_res and tables_res:
        answer = "Tham khảo hình ảnh và bảng sau:\n"
        # for img in images_res:
        #     answer += f"- Hình ảnh: {img}\n"
        # for table in tables_res:
        #     answer += f"- Bảng: {table}\n"
    elif images_res:
        answer = "Tham khảo hình ảnh sau:\n"
        # for img in images_res:
        #     answer += f"- Hình ảnh: {img}\n"
    elif tables_res:
        answer = "Tham khảo bảng sau:\n"
        # for table in tables_res:
        #     answer += f"- Bảng: {table}\n"

    return answer, images_res, tables_res, zip(text_doc, image_doc, table_doc), zip(text_metadata, image_metadata, table_metadata), zip(text_distances, image_distances, table_distances)
