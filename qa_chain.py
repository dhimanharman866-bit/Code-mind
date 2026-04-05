import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

def build_qa_chain(retriever):
    # Try Streamlit secrets first, then environment variable
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("GROQ_API_KEY not found! Add it to Codespaces secrets or Streamlit secrets.")
        st.stop()

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
You are an expert software engineer analyzing a codebase.

Conversation History:
{chat_history}

Retrieved Code Context:
{context}

Question: {question}

Instructions:
- Answer ONLY using the code context provided above
- Always start your answer by mentioning the file name like: "In `filename.py`:"
- Identify the programming language in one word at the start
- Be direct and specific — no vague summaries
- If a bug is found, show the broken code and then the fixed version
- If the answer is not in the context, respond exactly: "Not found in codebase"
- Format code blocks using markdown backticks
- You MUST mention every relevant file involved, not just the main ones
- Never mention README.md or config files unless directly asked

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(
            f"File: {doc.metadata.get('source')}\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        {
            "context": RunnableLambda(lambda x: x.get('question','')) | retriever | format_docs,
            "question": RunnableLambda(lambda x: x.get('question','')),
            "chat_history": RunnableLambda(lambda x: x.get('chat_history',''))
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain