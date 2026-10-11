import os
import streamlit as st
from rag_engine import InsuranceRAG

st.set_page_config(page_title="Insurance Policy Q&A", page_icon="📑", layout="wide")

# Cache the RAG engine so it stays in memory across interactions
@st.cache_resource
def get_rag_engine():
    return InsuranceRAG()

# Check for API key (from Streamlit Secrets or Environment Variables)
groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if not groq_key:
    st.error("GROQ_API_KEY is not set! Please add it in Streamlit app secrets.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_key

try:
    rag = get_rag_engine()
except Exception as e:
    st.error(f"Failed to initialize engine: {e}")
    st.stop()

st.title("📑 Insurance Policy Q&A Assistant")
st.caption("Powered by FastEmbed & Groq (Llama 3)")

with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload Policy PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Ingest Document", use_container_width=True):
            with st.spinner("Processing & indexing PDF..."):
                os.makedirs("./temp_uploads", exist_ok=True)
                temp_path = os.path.join("./temp_uploads", uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    chunk_count = rag.ingest_pdf(temp_path)
                    st.success(f"Ingested {chunk_count} chunks from `{uploaded_file.name}`!")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    st.divider()
    top_k = st.slider("Retrieval Top-K Chunks", min_value=1, max_value=8, value=5)

question = st.text_input("Ask a question about your policy:", placeholder="e.g., What is the waiting period for pre-existing diseases?")

if st.button("Search Policy", type="primary") and question:
    with st.spinner("Analyzing policy clauses..."):
        answer, sources = rag.generate_answer(question, top_k=top_k)

        st.markdown("### Answer")
        st.write(answer)

        if sources:
            st.markdown("---")
            st.subheader("Retrieved Citations & Evidence")
            for i, src in enumerate(sources):
                with st.expander(f"Clause {i+1}: {src['source']} (Page {src['page']}) — Similarity: {src['similarity']:.3f}"):
                    st.write(src["text"])