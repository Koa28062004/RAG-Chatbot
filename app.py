import streamlit as st
from mainV2 import generate_answer_with_source, ChromaDB
import pandas as pd

@st.cache_resource
def load_db():
    chroma_path = "chroma_db"
    text_db = ChromaDB.load_chroma_collection(chroma_path, name="text_docs")
    image_db = ChromaDB.load_chroma_collection(chroma_path, name="image_docs")
    table_db = ChromaDB.load_chroma_collection(chroma_path, name="table_docs")
    return text_db, image_db, table_db

# Load the databases
text_db, image_db, table_db = load_db()

st.title("RAG Chatbot with Gemini and ChromaDB")

query = st.text_input("Nhập câu hỏi của bạn:")

if st.button("Gửi câu hỏi") and query:
    with st.spinner("Đang tìm kiếm câu trả lời..."):
        answer, images_res, tables_res, docs, metadatas, distances = generate_answer_with_source(text_db, image_db, table_db, query=query)

    st.markdown("### Trả lời:")
    st.write(answer)
    if images_res:  
        st.markdown("Hình ảnh liên quan:")
        for i, image in enumerate(images_res):
            st.image(image, caption=f"Hình ảnh {i+1}", use_container_width=True)
    if tables_res:
        st.markdown("Bảng liên quan:")
        for i, table in enumerate(tables_res):
            try:
                if table.endswith(".csv"):
                    df = pd.read_csv(table)
                    st.dataframe(df)
                else:
                    with open(table, "r", encoding="utf-8") as f:
                        st.markdown(f.read(), unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Lỗi khi tải bảng từ {table}: {e}")

    st.markdown("---")
    st.markdown("### Các đoạn văn bản liên quan được tìm thấy (20 đoạn đầu tiên):")

    for i, ((text_chunk, image_chunk, table_chunk),
        (text_meta, image_meta, table_meta),
        (text_dist, image_dist, table_dist)) in enumerate(zip(docs, metadatas, distances)):

        if i >= 30:
            break

        # TEXT CHUNK
        if text_chunk.strip():
            source = text_meta.get("filename", "Không rõ nguồn")
            st.markdown(f"**[Text {i+1}] - nguồn:** {source} - sự khác biệt: {text_dist:.4f}")
            st.write(text_chunk)
            st.markdown("---")

        # IMAGE CHUNK
        if image_chunk.strip():
            source = image_meta.get("url", "Không rõ nguồn")
            st.markdown(f"**[Image {i+1}] - nguồn:** {source} - sự khác biệt: {image_dist:.4f}")
            try:
                st.image(source, caption=image_chunk, use_container_width=True)
            except Exception as e:
                st.warning(f"Lỗi khi hiển thị hình ảnh {source}: {e}")
            st.markdown("---")

        # TABLE CHUNK
        if table_chunk.strip():
            source = table_meta.get("url", "Không rõ nguồn")
            st.markdown(f"**[Table {i+1}] - nguồn:** {source} - sự khác biệt: {table_dist:.4f}")
            try:
                if source.endswith(".csv"):
                    df = pd.read_csv(source)
                    st.dataframe(df)
                else:
                    with open(source, "r", encoding="utf-8") as f:
                        st.markdown(f.read(), unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Lỗi khi tải bảng từ {source}: {e}")
            st.markdown("---")


