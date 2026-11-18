import os
import hashlib
from typing import List

import streamlit as st
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_astradb import AstraDBVectorStore


st.set_page_config(page_title="PDF Chatbot", page_icon="📚", layout="wide")
st.title("📚 Chat with your PDF")
st.caption("Upload a PDF, index its content in Astra DB, and ask questions using retrieval-augmented generation.")

def get_secret(name: str, default: str = "") -> str:
    """Read a secret from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return value or os.getenv(name, default)

def required_config():
    names = [
        "OPENAI_API_KEY",
        "ASTRA_DB_APPLICATION_TOKEN",
        "ASTRA_DB_API_ENDPOINT",
    ]
    missing = [name for name in names if not get_secret(name)]
    return missing

def get_vector_store():
    embedding = OpenAIEmbeddings(
        model=get_secret("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=get_secret("OPENAI_API_KEY"),
    )
    return AstraDBVectorStore(
        collection_name=get_secret("ASTRA_DB_COLLECTION", "pdf_chatbot_documents"),
        embedding=embedding,
        api_endpoint=get_secret("ASTRA_DB_API_ENDPOINT"),
        token=get_secret("ASTRA_DB_APPLICATION_TOKEN"),
    )

def extract_pdf(uploaded_file) -> List[Document]:
    reader = PdfReader(uploaded_file)
    docs = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": uploaded_file.name, "page": page_number},
            ))
    return docs

def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=180,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)

def answer_question(question: str, vector_store, k: int = 5) -> tuple[str, list]:
    matches = vector_store.similarity_search(question, k=k)
    if not matches:
        return "I couldn't find relevant information in the indexed PDF documents.", []

    context_parts = []
    for doc in matches:
        source = doc.metadata.get("source", "Uploaded PDF")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[Source: {source}, page {page}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    llm = ChatOpenAI(
        model=get_secret("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        api_key=get_secret("OPENAI_API_KEY"),
        temperature=0,
    )
    prompt = f"""You are a helpful PDF question-answering assistant.
Answer the user's question using only the context below.
If the answer is not in the context, say that the uploaded documents do not provide enough information.
Do not invent facts. Include source filenames and page numbers where possible.

CONTEXT:
{context}

QUESTION:
{question}
"""
    response = llm.invoke(prompt)
    return str(response.content), matches

missing = required_config()
with st.sidebar:
    st.header("Configuration")
    st.write("This app uses OpenAI for embeddings and answers, and Astra DB for vector storage.")
    st.caption("Add credentials in Streamlit Cloud → App settings → Secrets, or in `.streamlit/secrets.toml` locally.")
    st.divider()
    st.write("**Collection:**", get_secret("ASTRA_DB_COLLECTION", "pdf_chatbot_documents"))
    st.write("**Chat model:**", get_secret("OPENAI_CHAT_MODEL", "gpt-4o-mini"))

if missing:
    st.error("Missing configuration: " + ", ".join(missing))
    st.info("Add the required secrets before using the app. See README.md for the exact format.")
    st.stop()

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("1. Upload and index PDFs")
uploads = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)
if st.button("Index uploaded PDFs", type="primary", disabled=not uploads):
    try:
        with st.spinner("Reading PDFs and creating searchable embeddings..."):
            store = get_vector_store()
            total_chunks = 0
            for uploaded in uploads:
                file_bytes = uploaded.getvalue()
                digest = hashlib.sha256(file_bytes).hexdigest()
                if digest in st.session_state.indexed_files:
                    continue
                # Rewind the in-memory upload before parsing.
                uploaded.seek(0)
                page_docs = extract_pdf(uploaded)
                if not page_docs:
                    st.warning(f"No extractable text found in {uploaded.name}. It may be a scanned PDF; OCR is not included.")
                    continue
                chunks = split_documents(page_docs)
                store.add_documents(chunks)
                st.session_state.indexed_files.add(digest)
                total_chunks += len(chunks)
        if total_chunks:
            st.success(f"Indexed {total_chunks} text chunks from the selected PDFs.")
        else:
            st.info("No new text chunks were added. These files may already be indexed in this session.")
    except Exception as exc:
        st.error(f"Indexing failed: {exc}")

st.divider()
st.subheader("2. Ask questions")
if not st.session_state.indexed_files:
    st.info("Upload and index at least one PDF in this session before asking questions.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about your uploaded PDFs...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents and generating an answer..."):
                try:
                    store = get_vector_store()
                    answer, sources = answer_question(question, store)
                    st.markdown(answer)
                    if sources:
                        with st.expander("Retrieved source passages"):
                            for index, doc in enumerate(sources, start=1):
                                st.markdown(
                                    f"**{index}. {doc.metadata.get('source', 'PDF')} — page {doc.metadata.get('page', '?')}**"
                                )
                                st.write(doc.page_content[:1200])
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as exc:
                    error_message = f"Unable to answer this question: {exc}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

st.caption("Educational document assistant. Answers can be incomplete; verify important information against the original PDF.")
