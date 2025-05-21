from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def make_rag_prompt(query, context: str):
    prompt = f"""Bạn là trợ lý ảo chăm sóc khách hàng. Dưới đây là một số đoạn văn bản liên quan. Vui lòng đọc kỹ và đưa ra câu trả lời cho câu hỏi sau một cách lịch sự, chu đáo và chính xác. 

- Nếu bạn tìm thấy thông tin phù hợp, hãy trả lời đầy đủ, dễ hiểu và không tự suy diễn thêm ngoài những gì đã có.
- Nếu không tìm thấy thông tin liên quan, hãy đề xuất khách hàng liên hệ với nhân viên tư vấn để được hỗ trợ thêm.

Câu trả lời cần được định dạng theo JSON như sau:

{{
"response": <nội dung câu trả lời>,
"needAgent": <True nếu cần nhân viên hỗ trợ, False nếu đã đủ thông tin>
}}

Đoạn văn bản:
{context}

Câu hỏi:
{query}
"""
    return prompt

def generate_gemini_answer(prompt: str):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return response.text


def generate_answer_with_source(text_db, image_db, table_db, query, text_n_results=25, image_n_results=3, table_n_results=3):
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

    return (
    answer,
    images_res,
    tables_res,
    list(zip(text_doc, text_metadata, text_distances)),  # Convert zip to list
    list(zip(image_doc, image_metadata, image_distances)),  # Convert zip to list
    list(zip(table_doc, table_metadata, table_distances))  # Convert zip to list
)
