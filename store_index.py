from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

print("🚀 RUNNING FILE:", __file__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Load + process data
docs = load_pdf_file("data/")
print(f"Loaded {len(docs)} documents")

docs = filter_to_minimal_docs(docs)
chunks = text_split(docs)
print(f"Created {len(chunks)} chunks")

# Embeddings
embedding = download_hugging_face_embeddings()

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    print("⚡ Creating index...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("✅ Index created")

# Upload
PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embedding,
    index_name=index_name
)

print("✅ Data uploaded successfully!")