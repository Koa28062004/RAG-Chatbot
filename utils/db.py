import os
import json
from typing import List
import chromadb
from utils.embeddingFunction import SentenceTransformerEmbeddingFunction, GeminiEmbeddingFunction
from tqdm import tqdm   

# ChromaDB wrapper
class ChromaDB:
    @staticmethod
    def create_chroma_db(documents: List[dict], path: str, name: str, batch_size: int = 50, embedding_fn: str = "vn-law-embedding"):
        chroma_client = chromadb.PersistentClient(path=path)
        if embedding_fn == "vn-law-embedding":
            embedding_fn = SentenceTransformerEmbeddingFunction()
        elif embedding_fn == "gemini":
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
    def load_chroma_collection(path: str, name: str, embedding_fn: str = "vn-law-embedding"):
        if embedding_fn == "vn-law-embedding":
            embedding_fn = SentenceTransformerEmbeddingFunction()
        elif embedding_fn == "gemini":
            embedding_fn = GeminiEmbeddingFunction()
        chroma_client = chromadb.PersistentClient(path=path)
        return chroma_client.get_collection(name=name, embedding_function=embedding_fn)