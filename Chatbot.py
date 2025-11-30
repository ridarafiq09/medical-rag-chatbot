import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough



# Load environment variables
load_dotenv()



# 1. Load FAISS Vectorstore (NO CACHE)

def load_vectorstore():
    DB_FAISS_PATH = "vectorstore/db_faiss"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db



# 2. Load LLM (NO CACHE)
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=512,
        api_key=os.environ["GROQ_API_KEY"],
    )



# 3. RAG Prompt 
prompt = ChatPromptTemplate.from_template("""
You are a helpful medical assistant.

Use ONLY the following retrieved context to answer the question.
If the answer is not found in the context, say:
"I don't know based on the provided information."

CONTEXT:
{context}

QUESTION:
{input}

Answer:
""")



# 4. Build RAG Chain 
def build_rag_chain():
    db = load_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    llm = load_llm()

    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain, retriever



# Greeting Detector

GREETINGS = ["hi", "hello", "hey", "hola", "yo", "hiya"]

def is_greeting(text):
    text = text.lower().strip()
    first = text.split()[0]
    return first in GREETINGS



# 5. Streamlit UI
def main():

    st.title("🩺 Medical RAG Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("Ask your medical question...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Greeting handler
        if is_greeting(user_input):
            reply = "Hello! I'm your medical assistant. How can I help you today?"
            st.chat_message("assistant").markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            return

        try:
            rag_chain, retriever = build_rag_chain()

            # NO rewriting — use your exact question
            question = user_input

            # Run RAG
            response = rag_chain.invoke(question)
            answer = response.content

            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Show source documents
            with st.expander("Source Documents"):
                docs = retriever.invoke(question)
                for doc in docs:
                    st.markdown(f"**Source:** {doc.metadata}")
                    st.write(doc.page_content[:500] + "...\n")

        except Exception as e:
            st.error(f"Error: {str(e)}")



# Run App

if __name__ == "__main__":
    main()
