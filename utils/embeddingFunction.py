from typing import List
import os
import json
from tqdm import tqdm
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# SentenceTransformer embedding function
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("truro7/vn-law-embedding", truncate_dim=128)

    def __call__(self, inputs: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(inputs, convert_to_tensor=True)
        return embeddings.tolist()

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