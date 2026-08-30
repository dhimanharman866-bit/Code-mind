import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
from retriever import get_all_sources

load_dotenv()

def format_chat_history(chat_history):
    """Convert [(human, ai), ...] tuples to readable string."""
    if not chat_history:
        return "No previous conversation."
    return "\n".join(f"Human: {h}\nAssistant: {a}" for h, a in chat_history)

def build_qa_chain(retriever, vector_store=None):
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("GROQ_API_KEY not found! Add it to Codespaces secrets or Streamlit secrets.")
        st.stop()

    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0,
        max_tokens=4096,
    )

    # Get all file paths for context
    all_files = get_all_sources(vector_store) if vector_store else []
    file_list_context = "\n".join(f"- {f}" for f in all_files) if all_files else "Unknown"

    prompt = ChatPromptTemplate.from_template("""You are a senior software engineer. Provide thorough, well-structured answers using ONLY the provided code context.

PROJECT FILES (complete list):
{file_list}

CODE CONTEXT (relevant chunks):
{context}

CHAT HISTORY:
{chat_history}

QUESTION: {question}

REQUIREMENTS:
1. Use ONLY information from CODE CONTEXT above.
2. If answer not in context, say exactly: "Not found in codebase"
3. Cite relevant file names for every claim.
4. For questions about project structure, file count, or file listing, use the PROJECT FILES list above.
5. Provide DETAILED, STRUCTURED responses:
   - **Summary**: 2-3 sentence high-level answer
   - **Key Files**: List all relevant files with their roles
   - **Detailed Explanation**: Walk through the code logic, functions, classes, data flow
   - **Code Evidence**: Quote exact relevant code snippets
   - **Dependencies/Interactions**: How this connects to other parts

FORMAT YOUR RESPONSE WITH CLEAR SECTIONS AND BULLET POINTS.

ANSWER:""")

    def format_docs(docs):
        return "\n\n".join(
            f"File: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        {
            "context": RunnableLambda(lambda x: x.get('question', '')) | retriever | format_docs,
            "question": RunnableLambda(lambda x: x.get('question', '')),
            "chat_history": RunnableLambda(lambda x: format_chat_history(x.get('chat_history', []))),
            "file_list": RunnableLambda(lambda _: file_list_context),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain