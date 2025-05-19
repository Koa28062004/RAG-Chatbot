from scanPdf2Md import convert_pdf_to_markdown, extract_images_and_caption_flexible, extract_tables_from_markdown
import os
import json
from codeFunction import ChromaDB, load_text_documents, load_json_documents, generate_answer_with_source

if __name__ == "__main__":
    pdf_folder = "data"
    md_folder = "data-md"
    image_folder = "output-images"
    image_doc_json = "image_doc.json"
    table_doc_path = "table_doc.json"
    chroma_path = "chroma_db_gemini"
    embedding_fn = "gemini"

    # convert_pdf_to_markdown(pdf_folder, md_folder)
    
    # all_image_docs = []
    # for filename in os.listdir(pdf_folder):
    #     if filename.endswith(".pdf"):
    #         pdf_path = os.path.join(pdf_folder, filename)
    #         image_docs = extract_images_and_caption_flexible(pdf_path, md_folder, image_folder)
    #         if image_docs:
    #             all_image_docs.extend(image_docs)
    #             print(f"Image documents saved to {image_doc_json}")
    # with open(image_doc_json, "w", encoding="utf-8") as f:
    #     json.dump(all_image_docs, f, ensure_ascii=False, indent=4)  
    
    # extract_tables_from_markdown(md_folder, "output-tables", table_doc_path)

    text_docs = load_text_documents(pdf_folder)
    image_docs = load_json_documents(image_doc_json, "image")
    table_docs = load_json_documents(table_doc_path, "table")

    print("📂 Loaded:")
    print(f" - {len(text_docs)} text documents")
    print(f" - {len(image_docs)} image captions")
    print(f" - {len(table_docs)} table captions")

    # Create three separate collections
    text_db = ChromaDB.create_chroma_db(text_docs, chroma_path, name="text_docs", embedding_fn=embedding_fn)
    image_db = ChromaDB.create_chroma_db(image_docs, chroma_path, name="image_docs", embedding_fn=embedding_fn)
    table_db = ChromaDB.create_chroma_db(table_docs, chroma_path, name="table_docs", embedding_fn=embedding_fn)

    # text_db = ChromaDB.load_chroma_collection(chroma_path, name="text_docs", embedding_fn=embedding_fn)
    # image_db = ChromaDB.load_chroma_collection(chroma_path, name="image_docs", embedding_fn=embedding_fn)
    # table_db = ChromaDB.load_chroma_collection(chroma_path, name="table_docs", embedding_fn=embedding_fn)

    question = "Vùng có nguy cơ gây lóa tạm thời đối với đường thoát nạn theo phương ngang"
    answer, images_res, tables_res, text_combined, image_combined, table_combined = generate_answer_with_source(text_db, image_db, table_db, query=question)
       
    print("\n>> Gemini Answer:\n", answer)