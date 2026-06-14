import streamlit as st
import requests
import json
import httpx

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="LexRAG — Legal Document Q&A",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ LexRAG — Production Legal RAG System")
st.caption("Indian Legal Document Retrieval and Q&A")

# ── Sidebar — System Status ───────────────────────────────────────────────────
with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        st.success(f"API: {health['status'].upper()}")
    except:
        st.error("API: OFFLINE — start uvicorn server")

    st.divider()
    st.caption("LexRAG v1.0.0")
    st.caption("Model: llama-3.3-70b-versatile")
    st.caption("Embeddings: BAAI/bge-base-en-v1.5")
    st.caption("Vector DB: Qdrant")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📄 Ingest Documents", "🔍 Query", "👤 Human Review"])


# ── Tab 1 — Document Ingestion ────────────────────────────────────────────────
with tab1:
    st.header("Upload Legal Documents")
    st.caption("Supported formats: PDF, TXT, DOCX")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a legal document",
            type=["pdf", "txt", "docx"],
        )
    with col2:
        doc_type = st.selectbox(
            "Document Type",
            ["act", "judgment", "constitution", "regulation", "generic"],
        )

    if uploaded_file and st.button("Ingest Document", type="primary"):
        with st.spinner("Ingesting document..."):
            try:
                response = requests.post(
                    f"{API_BASE}/ingest",
                    files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
                    data={"doc_type": doc_type},
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Document ingested successfully!")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Chunks Ingested", result["chunks_ingested"])
                    col2.metric("Document Type", result["doc_type"])
                    col3.metric("File Hash", result["file_hash"][:8] + "...")
                elif response.status_code == 409:
                    st.warning("Document already ingested — duplicate detected.")
                else:
                    st.error(f"Error: {response.json()}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")


# ── Tab 2 — Query ─────────────────────────────────────────────────────────────
with tab2:
    st.header("Ask a Legal Question")

    query = st.text_area(
        "Your question",
        placeholder="e.g. What is the punishment for theft under IPC?",
        height=100,
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        rewrite = st.toggle("Query Rewriting", value=True)
    with col2:
        doc_type_filter = st.selectbox(
            "Filter by document type (optional)",
            ["All", "act", "judgment", "constitution", "regulation"],
        )

    if st.button("Ask", type="primary") and query:
        payload = {
            "query": query,
            "rewrite": rewrite,
            "doc_type": None if doc_type_filter == "All" else doc_type_filter,
        }

        # Show rewritten query
        rewritten_placeholder = st.empty()

        # Stream the answer
        st.subheader("Answer")
        answer_placeholder = st.empty()
        sources_placeholder = st.empty()

        with st.spinner("Thinking..."):
            try:
                # Use non-streaming endpoint for simplicity in Streamlit
                response = requests.post(
                    f"{API_BASE}/query",
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()

                    if result.get("rewritten_query") and rewrite:
                        rewritten_placeholder.info(
                            f"🔄 Rewritten query: *{result['rewritten_query']}*"
                        )

                    answer_placeholder.markdown(result["answer"])

                    if result["sources"]:
                        with sources_placeholder.expander("📚 Sources", expanded=True):
                            for source in result["sources"]:
                                score = source.get("rerank_score", 0)
                                section = source.get("section", "")
                                doc = source.get("document", "")
                                st.markdown(
                                    f"**{doc}** — {section} "
                                    f"*(rerank score: {score:.2f})*"
                                )
                else:
                    st.error(f"Error: {response.json()}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure uvicorn is running.")
            except Exception as e:
                st.error(f"Error: {e}")


# ── Tab 3 — Human Review ──────────────────────────────────────────────────────
with tab3:
    st.header("Human Review Queue")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Refresh Queue"):
            st.rerun()

    try:
        pending = requests.get(f"{API_BASE}/review/pending", timeout=3).json()
        approved = requests.get(f"{API_BASE}/review/approved", timeout=3).json()

        col1, col2 = st.columns(2)
        col1.metric("Pending Reviews", len(pending))
        col2.metric("Approved (Golden Dataset)", len(approved))

        if not pending:
            st.info("No pending reviews. All caught up!")
        else:
            for item in pending:
                with st.expander(
                    f"🔍 [{item['review_id']}] {item['query']}", expanded=True
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Original Query**")
                        st.write(item["query"])
                        st.markdown("**Rewritten Query**")
                        st.write(item["rewritten_query"])

                    with col2:
                        st.markdown("**Retrieved Chunks**")
                        for chunk in item["retrieved_chunks"]:
                            st.caption(
                                f"{chunk.get('section', 'N/A')} "
                                f"(score: {chunk.get('rerank_score', 0):.2f})"
                            )
                            st.text(chunk["text"][:200] + "...")

                    st.markdown("**Draft Answer**")
                    st.info(item["draft_answer"])

                    st.markdown("**Your Decision**")
                    corrected = st.text_area(
                        "Corrected answer (leave empty to approve as-is)",
                        key=f"correction_{item['review_id']}",
                    )
                    note = st.text_input(
                        "Reviewer note",
                        key=f"note_{item['review_id']}",
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(
                            "✅ Approve",
                            key=f"approve_{item['review_id']}",
                            type="primary",
                        ):
                            response = requests.post(
                                f"{API_BASE}/review/decide",
                                json={
                                    "review_id": item["review_id"],
                                    "approved": True,
                                    "corrected_answer": corrected or None,
                                    "reviewer_note": note or None,
                                },
                            )
                            if response.status_code == 200:
                                st.success("Approved and added to golden dataset!")
                                st.rerun()

                    with col2:
                        if st.button(
                            "❌ Reject",
                            key=f"reject_{item['review_id']}",
                        ):
                            response = requests.post(
                                f"{API_BASE}/review/decide",
                                json={
                                    "review_id": item["review_id"],
                                    "approved": False,
                                    "reviewer_note": note or None,
                                },
                            )
                            if response.status_code == 200:
                                st.warning("Rejected.")
                                st.rerun()

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure uvicorn is running.")