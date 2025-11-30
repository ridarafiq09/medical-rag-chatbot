import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


# 1. LLM Setup

GROQ_API_KEY = os.environ["GROQ_API_KEY"]

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=512,
    api_key=GROQ_API_KEY,
)


# 2. Load FAISS Vector Database

DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})


# 3. LOCAL PROMPT (replaces langchainhub completely)

prompt = ChatPromptTemplate.from_template("""
You are a helpful medical assistant.

Use the following retrieved context to answer the user's question.

Context:
{context}

Question:
{input}

If the answer is not in the provided context, say:
"I don't know based on the provided information."
""")


# 4. RAG Runnable Chain

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
)


# 5. Run Query

user_query = input("Write Query Here: ")
response = rag_chain.invoke(user_query)

print("\nRESULT:\n")
print(response.content)

print("\nSOURCE DOCUMENTS:\n")

source_docs = retriever.invoke(user_query)

for doc in source_docs:
    print(f"- {doc.metadata} -> {doc.page_content[:200]}...")

