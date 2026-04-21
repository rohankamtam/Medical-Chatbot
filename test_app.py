from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load embeddings
embedding = download_hugging_face_embeddings()

# Load Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embedding
)

# Retriever
retriever = docsearch.as_retriever(search_kwargs={"k": 5})

# Groq LLM
chatModel = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

# Improved Prompt
system_prompt = (
    "You are a medical assistant chatbot.\n"
    "Use ONLY the provided context to answer the question.\n"
    "If the answer is not in the context, say 'I don't know'.\n\n"
    
    "Instructions:\n"
    "- Give clear and structured answers\n"
    "- Explain in simple terms\n"
    "- Include symptoms, causes, and treatment if relevant\n"
    "- Do NOT make up information\n\n"
    
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Format retrieved docs
def format_docs(docs):
    return "\n\n".join(
        f"Source: {d.metadata.get('source')}\n{d.page_content}"
        for d in docs
    )

# RAG pipeline
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | chatModel
)

# Routes
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"].strip()
    msg_lower = msg.lower()

    # Handle greetings
    if any(greet in msg_lower for greet in ["hi", "hello", "hey"]):
        return "Hello! Ask me any medical question."

    # Generate response
    response = rag_chain.invoke(msg)
    return str(response.content)


# Render-compatible run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Chatbot running on port {port}")
    app.run(host="0.0.0.0", port=port)