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
from src.database.crud import delete_document_record

def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

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

def sync_missing_chunks():
    """Self-healing function to re-chunk any document missing chunks in SQLite."""
    try:
        from src.database.models import DocumentChunk
        with SessionLocal() as db:
            docs = db.query(DocumentMetadata).all()
            for doc in docs:
                chunk_count = db.query(DocumentChunk).filter(DocumentChunk.doc_id == doc.doc_id).count()
                if chunk_count == 0 and doc.file_path and os.path.exists(doc.file_path):
                    pages = pdf_parser.extract_pages(doc.file_path, doc.doc_id, doc.file_name)
                    chunks = chunker.create_chunks(pages)
                    if chunks:
                        vector_store.add_chunks(chunks)
    except Exception as e:
        print(f"Sync missing chunks warning: {e}")

sync_missing_chunks()

# Header
st.title("📚 AI Research & Knowledge Assistant")
st.markdown("Enterprise-grade RAG Assistant for Document Processing, ML Domain Classification, Grounded Q&A with Citations, Summarization, and Analytics.")

# Sidebar Navigation
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Select Module", ["Document Management", "RAG Question Answering", "Summarize & Compare", "System Analytics"])

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

                with SessionLocal() as db:
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
    with SessionLocal() as db:
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

            st.markdown("---")
            st.subheader("🗑️ Delete Saved Document")

            doc_map = {f"{d.file_name} (Category: {d.category} | Chunks: {d.total_chunks}) [ID: {d.doc_id[:8]}]": d.doc_id for d in docs}
            selected_doc_label = st.selectbox("Select Document to Delete", options=list(doc_map.keys()), key="delete_doc_select")

            col_del, col_space = st.columns([2, 4])
            with col_del:
                if st.button("🗑️ Delete Selected Document", type="primary", key="btn_delete_selected_doc"):
                    doc_id_to_del = doc_map[selected_doc_label]
                    doc_obj = next((d for d in docs if d.doc_id == doc_id_to_del), None)
                    doc_name = doc_obj.file_name if doc_obj else doc_id_to_del

                    if delete_document_record(doc_id_to_del, db, vector_store):
                        st.success(f"Successfully deleted document **{doc_name}**!")
                        safe_rerun()
                    else:
                        st.error("Failed to delete document.")

            st.markdown("#### Document Actions")
            for doc in docs:
                with st.expander(f"📄 {doc.file_name} — Category: {doc.category} (ID: {doc.doc_id[:8]}...)", expanded=False):
                    c1, c2, c3 = st.columns([3, 3, 2])
                    with c1:
                        st.write(f"**Pages:** {doc.total_pages} | **Chunks:** {doc.total_chunks}")
                        st.write(f"**Status:** {doc.processing_status}")
                    with c2:
                        st.write(f"**Uploaded:** {doc.upload_timestamp.strftime('%Y-%m-%d %H:%M') if doc.upload_timestamp else 'N/A'}")
                        st.write(f"**Path:** `{doc.file_path}`")
                    with c3:
                        if st.button("🗑️ Delete Document", key=f"del_card_{doc.doc_id}", type="primary"):
                            if delete_document_record(doc.doc_id, db, vector_store):
                                st.success(f"Successfully deleted **{doc.file_name}**!")
                                safe_rerun()
                            else:
                                st.error(f"Failed to delete **{doc.file_name}**.")
        else:
            st.info("No documents stored in the knowledge base yet.")

elif menu == "RAG Question Answering":
    st.header("💬 Grounded Question Answering")
    
    with SessionLocal() as db:
        docs = db.query(DocumentMetadata).all()
    
    selected_doc_ids = []
    if docs:
        doc_map = {f"{d.file_name} [ID: {d.doc_id[:6]}] ({d.category})": d.doc_id for d in docs}
        selected = st.multiselect("Filter by Specific Document(s) (Optional)", list(doc_map.keys()))
        selected_doc_ids = [doc_map[s] for s in selected]

    query = st.text_input("Enter your research question:")
    if st.button("Get Answer") and query:
        with st.spinner("Retrieving relevant context and synthesizing grounded answer..."):
            result = rag_engine.answer_question(query=query, doc_ids=selected_doc_ids if selected_doc_ids else None)
            
            st.markdown("### Answer")
            st.write(result["answer"])

            st.markdown("### 📌 Citations")
            citations = result.get("citations", [])
            if citations:
                for cit in citations:
                    st.caption(f"• **Document:** {cit['document']} | **Page:** {cit['page']}")
            else:
                st.info("No explicit document citations returned.")

            with st.expander("View Retrieved Context Snippets"):
                retrieved = result.get("retrieved_context", [])
                if retrieved:
                    for idx, ctx in enumerate(retrieved):
                        st.markdown(f"**Chunk {idx+1}:**")
                        st.text(ctx)
                else:
                    st.write("No matching context chunks found.")

elif menu == "Summarize & Compare":
    st.header("🔍 Summarization & Comparison Engine")

    sub_tab1, sub_tab2 = st.tabs(["Document Summary", "Multi-Document Comparison"])

    with sub_tab1:
        with SessionLocal() as db:
            docs = db.query(DocumentMetadata).all()
        if docs:
            doc_options = {f"{d.file_name} [ID: {d.doc_id[:6]}]": d.doc_id for d in docs}
            selected_label = st.selectbox("Select Document to Summarize", list(doc_options.keys()))
            if st.button("Generate Summary"):
                doc_id = doc_options[selected_label]
                selected_name = selected_label.split(" [ID:")[0]
                summary = summarizer.summarize_document(doc_id, file_name=selected_name)
                
                st.markdown("### 📋 Executive Summary")
                st.info(summary.get("executive_summary"))

                st.markdown("### ⚙️ Technical Summary")
                st.write(summary.get("technical_summary"))

                st.markdown("### 📌 Key Bullet Points")
                for bp in summary.get("bullet_points", []):
                    st.markdown(f"• {bp}")

                st.markdown("### 💡 Key Takeaways")
                for kt in summary.get("key_takeaways", []):
                    st.markdown(f"• {kt}")

    with sub_tab2:
        with SessionLocal() as db:
            docs = db.query(DocumentMetadata).all()
        if docs and len(docs) >= 2:
            doc_options = {f"{d.file_name} [ID: {d.doc_id[:6]}]": d.doc_id for d in docs}
            selected_labels = st.multiselect("Select 2+ Documents to Compare", list(doc_options.keys()), default=list(doc_options.keys())[:2])
            if st.button("Compare Documents") and len(selected_labels) >= 2:
                selected_ids = [doc_options[n] for n in selected_labels]
                selected_names = [n.split(" [ID:")[0] for n in selected_labels]
                comp = comparator.compare_documents(selected_ids, doc_names=selected_names)
                
                st.markdown("### 🔬 Methodology Comparison")
                meth = comp.get("methodology_comparison", {})
                if isinstance(meth, dict):
                    for doc_n, m_desc in meth.items():
                        st.markdown(f"**{doc_n}:** {m_desc}")
                else:
                    st.write(str(meth))

                st.markdown("### 🤝 Similarities")
                for sim in comp.get("similarities", []):
                    st.markdown(f"• {sim}")

                st.markdown("### ⚡ Key Differences")
                for diff in comp.get("differences", []):
                    st.markdown(f"• {diff}")

                st.markdown("### 📊 Advantages & Characteristics")
                adv = comp.get("advantages_disadvantages", {})
                if isinstance(adv, dict):
                    for doc_n, a_info in adv.items():
                        st.markdown(f"**{doc_n}:**")
                        if isinstance(a_info, dict):
                            for adv_item in a_info.get("advantages", []):
                                st.caption(f"  + Advantage: {adv_item}")
                        else:
                            st.caption(f"  {a_info}")

elif menu == "System Analytics":
    st.header("📊 Knowledge Base System Analytics")
    with SessionLocal() as db:
        analytics = AnalyticsEngine(db, vector_store)
        stats = analytics.get_system_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Documents", stats.get("total_documents", 0))
    col2.metric("Total Pages", stats.get("total_pages", 0))
    col3.metric("Processed Chunks", stats.get("total_processed_chunks", 0))
    col4.metric("Questions Answered", stats.get("total_questions_answered", 0))

    st.markdown("### Category Distribution")
    st.bar_chart(stats.get("category_distribution", {}))

