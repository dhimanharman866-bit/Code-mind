import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
def build_qa_chain(retriever):
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
You are a senior software engineer.
                                              
Conversation History:
{chat_history}


Context:
{context}

Question:
{question}                                                                                                                                     
Rules:
- Answer ONLY from context
- Mention file name
- Identify language
- If bug → fix it
- If not found → say "Not found in codebase"


Answer:
""")

    # Format retrieved docs into text
    def format_docs(docs):
        return "\n\n".join(
            f"File: {doc.metadata.get('source')}\n{doc.page_content}"
            for doc in docs
        )

    # Runnable chain
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