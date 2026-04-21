# 🩺 Medical Chatbot (RAG-based)

A Retrieval-Augmented Generation (RAG) powered **Medical Chatbot** built using **LangChain, Pinecone, Groq (LLM), and Flask**.
This project answers medical queries based on uploaded PDF documents.

---

## 🚀 Features

* 📄 Extracts and processes medical PDFs
* 🔍 Semantic search using **Pinecone vector database**
* 🤖 Generates answers using **LLM (Groq - LLaMA 3.1)**
* 🌐 Interactive web UI using **Flask + HTML/CSS**
* 🧠 Context-aware responses (RAG pipeline)
* ⚡ Fully free stack (no OpenAI billing required)

---

## 🏗️ Tech Stack

* **Backend:** Flask
* **LLM:** Groq (LLaMA 3.1)
* **Vector DB:** Pinecone
* **Framework:** LangChain
* **Embeddings:** HuggingFace (MiniLM)
* **Frontend:** HTML, CSS, Bootstrap

---

## 📁 Project Structure

```
Medical-Chatbot/
│
├── app.py                # Flask app (main chatbot)
├── store_index.py        # Creates & uploads embeddings to Pinecone
├── src/
│   └── helper.py         # Data processing + embeddings
│
├── templates/
│   └── chat.html         # UI
│
├── static/
│   └── style.css         # Styling
│
├── data/                 # PDF files
├── .env                  # API keys
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repo

```
git clone https://github.com/your-username/medical-chatbot.git
cd medical-chatbot
```

---

### 2️⃣ Create environment

```
conda create -n medibot python=3.10 -y
conda activate medibot
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Add API keys

Create a `.env` file:

```
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
```

---

### 5️⃣ Add your PDFs

Place medical PDFs inside:

```
data/
```

---

## 🧠 Build Vector Index (ONE TIME)

```
python store_index.py
```

👉 This:

* Reads PDFs
* Splits into chunks
* Creates embeddings
* Uploads to Pinecone

---

## ▶️ Run the App

```
python app.py
```

Open in browser:

```
http://localhost:8080
```

---

## 💡 Example Queries

* What is diabetes?
* Symptoms of hypertension
* Treatment for acne
* Causes of asthma

---

## ⚠️ Notes

* This is a **demo project** (not for medical diagnosis)
* Answers depend on provided documents
* Free LLM may produce simpler responses

---

## 🚀 Future Improvements

* Add source citations in answers
* Improve UI/UX
* Deploy on AWS / Render
* Add conversation memory
* Use better LLM for higher accuracy

---

## 👨‍💻 Author

Rohan Kamtam

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
