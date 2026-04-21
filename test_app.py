from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

app = Flask(__name__)

load_dotenv()

# Load embedding
embedding = download_hugging_face_embeddings()

# Load Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embedding
)

retriever = docsearch.as_retriever(search_kwargs={"k": 5})

# Groq LLM
chatModel = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

# Prompt
system_prompt = (
    "Answer ONLY from the provided context. "
    "If the answer is not in the context, say 'I don't know'. "
    "Be concise.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | chatModel
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"].strip()
    msg_lower = msg.lower()

    if msg_lower in ["hi", "hello", "hey"]:
        return "Hello! Ask me any medical question."

    response = rag_chain.invoke(msg)
    return str(response.content)

if __name__ == "__main__":
    print("🚀 Chatbot running...")
    app.run(port=8080)