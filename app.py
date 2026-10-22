
import streamlit as st

from rag_engine import (
    extract_pdf_pages,
    create_chunks,
    build_search_index,
    search_documents,
    generate_answer,
)

st.set_page_config(
    page_title="Ask My PDF Bot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Ask My PDF Bot")
st.write(
    "Upload a PDF, ask questions about its contents, "
    "and receive AI-generated answers with sources."
)

# Upload documents
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

# Process uploaded documents
if uploaded_files:
    if st.button("Process Documents"):
        all_pages = []

        with st.spinner("Reading your documents..."):
            for file in uploaded_files:
                pages = extract_pdf_pages(
                    file.getvalue(),
                    file.name,
                )
                all_pages.extend(pages)

        chunks = create_chunks(all_pages)

        if not chunks:
            st.error(
                "No readable text was found. "
                "The PDF may be scanned or image-only."
            )
        else:
            with st.spinner(
                "Building the AI search index... "
                "This may take a little while the first time."
            ):
                index, chunks = build_search_index(chunks)

            st.session_state["search_index"] = index
            st.session_state["chunks"] = chunks
            st.session_state["processed_files"] = [
                file.name for file in uploaded_files
            ]
            st.session_state["chat_history"] = []

            st.success(
                f"Processed {len(uploaded_files)} PDF file(s) "
                f"into {len(chunks)} searchable chunks."
            )

# Ask questions after processing
if (
    "search_index" in st.session_state
    and st.session_state["search_index"] is not None
):
    st.divider()
    st.subheader("💬 Ask your documents")

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Enter your question",
            placeholder="Example: What topics does this document cover?",
        )
        submitted = st.form_submit_button("Get AI Answer")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Finding relevant passages..."):
                results = search_documents(
                    question,
                    st.session_state["search_index"],
                    st.session_state["chunks"],
                    top_k=4,
                )

            with st.spinner("Generating your answer..."):
                answer = generate_answer(question, results)

            st.session_state["chat_history"].append(
                {
                    "question": question,
                    "answer": answer,
                    "sources": results,
                }
            )

    # Display questions and answers
    for item in st.session_state.get("chat_history", []):
        st.markdown("### 🧑 Your question")
        st.write(item["question"])

        st.markdown("### 🤖 AI answer")
        st.markdown(item["answer"])

        if item["sources"]:
            with st.expander("View retrieved source passages"):
                for source in item["sources"]:
                    st.markdown(
                        f"**File:** {source['source']} "
                        f"· **Page:** {source['page']}"
                    )
                    st.write(source["text"])
                    st.divider()

    if st.button("Clear chat history"):
        st.session_state["chat_history"] = []
        st.rerun()

else:
    st.info(
        "Upload one or more PDFs and click "
        "'Process Documents' to get started."
    )
