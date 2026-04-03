import streamlit as st
import zipfile
import os
import tempfile
import subprocess
from loader import load_document
from chunker import split_document
from embeddings import get_embedding
from vectorstore import build_or_load_vectorstore
from retriever import get_retriever
from qa_chain import build_qa_chain

st.set_page_config(
    page_title="CodeMind",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #000000;
    color: #e0e0e0;
}

[data-testid="stSidebar"] {
    background-color: #0a0a0a;
    border-right: 1px solid #1a1a1a;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    color: #666666;
    border-radius: 0;
    padding: 8px 16px;
    border-bottom: 2px solid transparent;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ffffff;
    border-bottom: 2px solid #ffffff;
    background: transparent;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 2px solid #444444 !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 2px solid #ffffff !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background-color: #0d0d0d !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}

/* Text input */
[data-testid="stTextInput"] input {
    background-color: #0d0d0d !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed #2a2a2a;
    border-radius: 8px;
    padding: 20px;
    background: #0a0a0a;
    text-align: center;
}

/* Buttons */
.stButton button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 8px 16px !important;
    transition: opacity 0.2s !important;
}

.stButton button:hover {
    opacity: 0.85 !important;
}

/* Secondary button (Clear) */
.secondary-btn button {
    background-color: #1a1a1a !important;
    color: #e0e0e0 !important;
    border: 1px solid #2a2a2a !important;
}

/* Expander */
[data-testid="stExpander"] {
    background-color: #0a0a0a !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 8px !important;
}

/* Code blocks */
code {
    font-family: 'JetBrains Mono', monospace !important;
    background: #111111 !important;
    color: #c0c0c0 !important;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.82em;
}

pre {
    background: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 6px !important;
    padding: 12px !important;
}

pre code {
    background: transparent !important;
    padding: 0;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.status-ready {
    background: #0a1f0a;
    color: #4caf50;
    border: 1px solid #1a3a1a;
}
.status-idle {
    background: #111111;
    color: #555555;
    border: 1px solid #222222;
}

/* Metric cards */
.metric-card {
    background: #0d0d0d;
    border: 1px solid #1a1a1a;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 0.68rem;
    color: #555555;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Source pill */
.source-pill {
    display: inline-block;
    background: #111111;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    color: #aaaaaa;
    margin: 3px;
}

/* Divider */
hr {
    border-color: #1a1a1a !important;
}

/* Section labels */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #555555;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #000000; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────
for key, default in {
    "qa": None,
    "retriever": None,
    "messages": [],          # list of (role, text) tuples for display
    "chat_history": [],      # list of (human, ai) tuples for LangChain memory
    "doc_count": 0,
    "chunk_count": 0,
    "project_name": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def run_indexing(tmpdir, source_name):
    """Shared indexing logic for both ZIP and GitHub sources."""
    docs   = load_document(tmpdir)
    chunks = split_document(docs)

    if not docs:
        st.error("No supported source files found. Ensure the project contains .py / .js / .java / .cpp / .ts files.")
        st.stop()

    emb = get_embedding()
    vs  = build_or_load_vectorstore(chunks, emb)
    ret = get_retriever(vs)
    qa  = build_qa_chain(ret)

    st.session_state.qa           = qa
    st.session_state.retriever    = ret
    st.session_state.doc_count    = len(docs)
    st.session_state.chunk_count  = len(chunks)
    st.session_state.project_name = source_name
    st.session_state.messages     = []
    st.session_state.chat_history = []   # reset memory on new project


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='padding: 8px 0 4px 0'>
            <div style='font-size:1.1rem;font-weight:600;color:#ffffff;letter-spacing:0.02em'>CodeMind</div>
            <div style='font-size:0.75rem;color:#444444;margin-top:2px'>RAG-powered codebase assistant</div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Status
    if st.session_state.qa:
        st.markdown("<span class='status-badge status-ready'>Ready</span>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin-top:8px;font-size:0.8rem;color:#444444'>Project: <span style='color:#cccccc'>{st.session_state.project_name}</span></p>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-badge status-idle'>No project loaded</span>", unsafe_allow_html=True)

    st.divider()

    # Metrics
    if st.session_state.qa:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{st.session_state.doc_count}</div>
                <div class='metric-label'>Files</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{st.session_state.chunk_count}</div>
                <div class='metric-label'>Chunks</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("")
        st.divider()

    # Upload tabs
    st.markdown("<div class='section-label'>Load Project</div>", unsafe_allow_html=True)
    zip_tab, github_tab = st.tabs(["ZIP Upload", "GitHub Repo"])

    # ── ZIP tab ───────────────────────────────────────────────────────
    with zip_tab:
        uploaded_file = st.file_uploader(
            "Drop your ZIP here",
            type=["zip"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            if st.button("Index Codebase", use_container_width=True, key="zip_btn"):
                with st.spinner("Parsing and embedding..."):
                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            zip_path = os.path.join(tmpdir, "project.zip")
                            with open(zip_path, "wb") as f:
                                f.write(uploaded_file.read())
                            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                                zip_ref.extractall(tmpdir)
                            run_indexing(tmpdir, uploaded_file.name.replace(".zip", ""))
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

                st.success("Codebase indexed successfully.")
                st.rerun()

    # ── GitHub tab ────────────────────────────────────────────────────
    with github_tab:
        repo_url = st.text_input(
            "Repository URL",
            placeholder="https://github.com/user/repo",
            label_visibility="collapsed"
        )

        if repo_url:
            if st.button("Clone and Index", use_container_width=True, key="github_btn"):
                with st.spinner("Cloning repository..."):
                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            result = subprocess.run(
                                ["git", "clone", "--depth=1", repo_url, tmpdir],
                                capture_output=True,
                                text=True
                            )
                            if result.returncode != 0:
                                st.error(f"Clone failed: {result.stderr.strip()}")
                                st.stop()

                            repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
                            run_indexing(tmpdir, repo_name)

                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

                st.success("Repository indexed successfully.")
                st.rerun()

    st.divider()

    # Clear chat
    if st.session_state.messages:
        with st.container():
            if st.button("Clear Conversation", use_container_width=True, key="clear_btn"):
                st.session_state.messages     = []
                st.session_state.chat_history = []   # also clear LangChain memory
                st.rerun()

    st.markdown("<p style='color:#222222;font-size:0.68rem;margin-top:16px'>Powered by LangChain + Groq</p>", unsafe_allow_html=True)


# ── Main chat area ────────────────────────────────────────────────────
st.markdown("""
    <div style='padding: 8px 0 24px 0; border-bottom: 1px solid #1a1a1a; margin-bottom: 24px'>
        <div style='font-size:1.4rem;font-weight:600;color:#ffffff'>Chat</div>
        <div style='font-size:0.78rem;color:#444444;margin-top:2px'>Ask questions about your indexed codebase</div>
    </div>
""", unsafe_allow_html=True)

if not st.session_state.qa:
    st.markdown("""
    <div style='text-align:center;padding:100px 20px;'>
        <div style='font-size:0.95rem;color:#333333;font-weight:500'>No project loaded</div>
        <div style='font-size:0.8rem;color:#2a2a2a;margin-top:8px'>Upload a ZIP file or clone a GitHub repository to get started</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Render existing messages
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)

# Chat input
query = st.chat_input("Ask anything about your codebase...")

if query:
    # 1. Show and store the user message
    st.session_state.messages.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Call the QA chain, passing the stored LangChain-format chat history
    with st.chat_message("assistant"):
        with st.spinner(""):
            result = st.session_state.qa.invoke({
                "question": query,
                "chat_history": st.session_state.chat_history   # list of (human, ai) tuples
            })

            # ConversationalRetrievalChain returns a dict with key "answer"
            answer = result.get("answer", result) if isinstance(result, dict) else result
            st.markdown(answer)

    # 3. Save both display message and LangChain memory tuple
    st.session_state.messages.append(("assistant", answer))
    st.session_state.chat_history.append((query, answer))   # (human, ai) for LangChain

    # 4. Show source references
    if st.session_state.retriever:
        source_docs    = st.session_state.retriever.invoke(query)
        unique_sources = list({doc.metadata.get("source", "unknown") for doc in source_docs})

        with st.expander(f"{len(unique_sources)} source file(s) referenced"):
            for src in unique_sources:
                fname = os.path.basename(src)
                st.markdown(f"<span class='source-pill'>{fname}</span>", unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            for doc in source_docs:
                fname = os.path.basename(doc.metadata.get("source", "unknown"))
                st.markdown(f"**`{fname}`**")
                st.code(
                    doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                    language="python"
                )