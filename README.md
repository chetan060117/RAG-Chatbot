# RAG-Chatbot
rag based Whatsapp Chatbot

# PureDrop Filters WhatsApp Chatbot

A Flask-based, RAG-powered WhatsApp chatbot that lets customers ask questions about PureDrop Filters product documentation and download PDF reports on demand. Built with free-tier and community-licensed tooling: LangChain, FAISS, Hugging Face, Twilio Sandbox, and ngrok.

---

## 🔍 Features

1. **Contextual Q&A**  
   - Ingests a PDF product manual (`filter.pdf`) via `PyPDFLoader`  
   - Splits text into chunks (800 chars) with `RecursiveCharacterTextSplitter`  
   - Embeds chunks using `HuggingFaceEmbeddings (all-MiniLM-L6-v2)` stored in FAISS  
   - Retrieves top-k relevant chunks and feeds them, via LangChain, into a Hugging Face Zephyr-7B LLM  
   - Enforces “answer only from context” with a prompt template and fallback for out-of-scope queries

2. **Interactive “Get Report”**  
   - Lists all PDF files in the `Reports/` directory when user types `get report`  
   - Sends the chosen report as a WhatsApp media attachment on `report <number>`  
   - Serves files securely via a Flask endpoint at `/report/<id>`

3. **Twilio & ngrok Integration**  
   - Uses Twilio MessagingResponse to handle WhatsApp webhook requests  
   - Exposes localhost securely via ngrok for development and Twilio sandbox compatibility  
   - Keeps all keys (Twilio, Hugging Face) in a `.env` file loaded with python-dotenv

4. **Robust & Modular Architecture**  
   - Single Flask app (`app.py`) with separate routes for health check, `/whatsapp`, and file serving  
   - Modular LangChain setup: document loading → vector store → retriever → prompt → LLM chain  
   - Easily swap in other embedding models, vector stores, or LLM endpoints

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+  
- A Twilio account (WhatsApp Sandbox enabled)  
- ngrok (for tunneling your localhost)  
- A Hugging Face API token (if needed for the Zephyr-7B endpoint)

## Project Structure

.
├── Reports/                  # PDF reports available via “get report”
├── filter.pdf                # Main product manual for Q&A
├── app.py                    # Flask application and LangChain logic
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables 
└── README.md                 # This file

## 🤝 Contributing
Feel free to open issues or submit pull requests to improve features, swap in new models, or add test coverage.

## 📜 License
This project is licensed under the MIT License.
