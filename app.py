import streamlit as st
from main import load_chroma_collection, generate_answer_with_source

@st.cache_resource
def load_db():
    db_path = "db"
    collection_name = "rag_experiment"
    return load_chroma_collection(path=db_path, name=collection_name)

db = load_db()

st.title("RAG Chatbot with Gemini and ChromaDB")

query = st.text_input("Nhập câu hỏi của bạn:")

if st.button("Gửi câu hỏi") and query:
    with st.spinner("Đang tìm kiếm câu trả lời..."):
        answer, docs, metadatas = generate_answer_with_source(db, query)

    st.markdown("### Trả lời:")
    st.write(answer)

    st.markdown("---")
    st.markdown("### Các đoạn văn bản liên quan được tìm thấy (10 đoạn đầu tiên):")
    for i, (doc, meta) in enumerate(zip(docs[:10], metadatas[:10])):
        source = meta.get("source_file", "Không rõ nguồn")
        st.markdown(f"**Đoạn {i+1} - nguồn:** {source}")
        st.write(doc)
        st.markdown("---")

