import os
import uuid
import streamlit as st
import pandas as pd
from datetime import datetime

from config.settings import settings
from src.database.base import Base, engine, SessionLocal
from src.database.models import DocumentMetadata, ChatSession
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.chunker import DocumentChunker
from src.ml.predictor import DocumentClassifier
from src.vector_store.manager import VectorStoreManager
from src.rag.qa_chain import RAGEngine
from src.rag.summarizer import DocumentSummarizer
from src.rag.comparator import DocumentComparator
from src.analytics.metrics import AnalyticsEngine

# Page configuration
st.set_page_config(
    page_title="AI Research & Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize DB & Core Engines
Base.metadata.create_all(bind=engine)
pdf_parser = PDFParser()
chunker = DocumentChunker()
classifier = DocumentClassifier()
vector_store = VectorStoreManager()
rag_engine = RAGEngine(vector_store)
summarizer = DocumentSummarizer(vector_store)
comparator = DocumentComparator(vector_store)

# Header
st.title("📚 AI Research & Knowledge Assistant")
st.markdown("Enterprise-grade RAG Assistant for Document Processing, ML Domain Classification, Grounded Q&A with Citations, Summarization, and Analytics.")

# Sidebar Navigation
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Select Module", ["Document Management", "RAG Question Answering", "Summarize & Compare", "System Analytics"])

db = SessionLocal()

if menu == "Document Management":
    st.header("📄 Upload & Ingest Documents")
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Extracting pages, chunking, and classifying via ML..."):
                doc_id = str(uuid.uuid4())
                save_path = os.path.join(settings.RAW_DOCUMENTS_DIR, f"{doc_id}_{uploaded_file.name}")
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                pages = pdf_parser.extract_pages(save_path, doc_id, uploaded_file.name)
                full_text = " ".join([p["text"] for p in pages])
                prediction = classifier.predict(full_text)
                predicted_category = prediction.get("category", "Unclassified")

                chunks = chunker.create_chunks(pages)
                vector_store.add_chunks(chunks)

                doc_record = DocumentMetadata(
                    doc_id=doc_id,
                    file_name=uploaded_file.name,
                    file_path=save_path,
                    upload_timestamp=datetime.utcnow(),
                    total_pages=len(pages),
                    total_chunks=len(chunks),
                    processing_status="PROCESSED",
                    category=predicted_category
                )
                db.add(doc_record)
                db.commit()

                st.success(f"Successfully processed **{uploaded_file.name}**!")
                st.info(f"**Pages:** {len(pages)} | **Chunks:** {len(chunks)} | **ML Category:** {predicted_category}")

    st.subheader("Indexed Knowledge Base Documents")
    docs = db.query(DocumentMetadata).all()
    if docs:
        doc_data = [{
            "Document ID": d.doc_id,
            "File Name": d.file_name,
            "Pages": d.total_pages,
            "Chunks": d.total_chunks,
            "Category": d.category,
            "Status": d.processing_status
        } for d in docs]
        st.dataframe(pd.DataFrame(doc_data), use_container_width=True)

elif menu == "RAG Question Answering":
    st.header("💬 Grounded Question Answering")
    
    docs = db.query(DocumentMetadata).all()
    selected_doc_ids = []
    if docs:
        doc_map = {f"{d.file_name} ({d.category})": d.doc_id for d in docs}
        selected = st.multiselect("Filter by Specific Document(s) (Optional)", list(doc_map.keys()))
        selected_doc_ids = [doc_map[s] for s in selected]

    query = st.text_input("Enter your research question:")
    if st.button("Get Answer") and query:
        with st.spinner("Retrieving relevant context and synthesizing grounded answer..."):
            result = rag_engine.answer_question(query=query, doc_ids=selected_doc_ids if selected_doc_ids else None)
            
            st.markdown("### Answer")
            st.write(result["answer"])

            st.markdown("### 📌 Citations")
            for cit in result.get("citations", []):
                st.caption(f"• **Document:** {cit['document']} | **Page:** {cit['page']}")

            with st.expander("View Retrieved Context Snippets"):
                for idx, ctx in enumerate(result.get("retrieved_context", [])):
                    st.markdown(f"**Chunk {idx+1}:**")
                    st.text(ctx)

elif menu == "Summarize & Compare":
    st.header("🔍 Summarization & Comparison Engine")

    sub_tab1, sub_tab2 = st.tabs(["Document Summary", "Multi-Document Comparison"])

    with sub_tab1:
        docs = db.query(DocumentMetadata).all()
        if docs:
            doc_options = {d.file_name: d.doc_id for d in docs}
            selected_name = st.selectbox("Select Document to Summarize", list(doc_options.keys()))
            if st.button("Generate Summary"):
                doc_id = doc_options[selected_name]
                summary = summarizer.summarize_document(doc_id, file_name=selected_name)
                
                st.markdown("### Executive Summary")
                st.write(summary.get("executive_summary"))

                st.markdown("### Technical Summary")
                st.write(summary.get("technical_summary"))

                st.markdown("### Key Bullet Points")
                for bp in summary.get("bullet_points", []):
                    st.write(f"• {bp}")

    with sub_tab2:
        docs = db.query(DocumentMetadata).all()
        if docs and len(docs) >= 2:
            doc_options = {d.file_name: d.doc_id for d in docs}
            selected_names = st.multiselect("Select 2+ Documents to Compare", list(doc_options.keys()), default=list(doc_options.keys())[:2])
            if st.button("Compare Documents") and len(selected_names) >= 2:
                selected_ids = [doc_options[n] for n in selected_names]
                comp = comparator.compare_documents(selected_ids, doc_names=selected_names)
                
                st.markdown("### Similarities")
                for sim in comp.get("similarities", []):
                    st.write(f"• {sim}")

                st.markdown("### Key Differences")
                for diff in comp.get("differences", []):
                    st.write(f"• {diff}")

elif menu == "System Analytics":
    st.header("📊 Knowledge Base System Analytics")
    analytics = AnalyticsEngine(db, vector_store)
    stats = analytics.get_system_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Documents", stats.get("total_documents", 0))
    col2.metric("Total Pages", stats.get("total_pages", 0))
    col3.metric("Processed Chunks", stats.get("total_processed_chunks", 0))
    col4.metric("Questions Answered", stats.get("total_questions_answered", 0))

    st.markdown("### Category Distribution")
    st.bar_chart(stats.get("category_distribution", {}))

db.close()
