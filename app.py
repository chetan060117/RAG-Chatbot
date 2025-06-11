import os
from dotenv import load_dotenv
from flask import Flask, request, send_file
from twilio.twiml.messaging_response import MessagingResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return "Flask WhatsApp chatbot is running."
    return whatsapp_reply()

# Load and preprocess document (ensure the "filter.pdf" path is correct)
loader = PyPDFLoader("filter.pdf")
raw_docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
doc_chunks = splitter.split_documents(raw_docs)

# Create vector store using updated embeddings and FAISS
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(doc_chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Set up LLM (ensure your Hugging Face token is set in the environment if required)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    temperature=0.2
)

# Define prompt template
prompt = PromptTemplate(
    template="""
You are a helpful assistant who strictly answers questions only about PureDrop Filters.
Answer ONLY using the context provided below. Do NOT include any external information.
If the provided context is insufficient or if the question is not related to PureDrop Filters,
respond exactly with: "I can't answer that question, ask me anything about PureDrop Filters."

Context:
{context}

Question:
{question}
    """,
    input_variables=["context", "question"]
)

# Chain setup: retrieve documents, format them, build the prompt, generate the answer, and parse output.
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser

# WhatsApp webhook to handle incoming messages.
@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.form.get("Body", "").strip().lower()
    resp = MessagingResponse()

    greetings = {"hi", "hello", "hey", "greetings"}

    if set(incoming_msg.split()).intersection(greetings):
        resp.message(
            "Hello there! Welcome to the PureDrop Filters chatbot. "
            "How can I help you today?\n\nType your question, or type 'get report' to view available reports."
        )
    elif incoming_msg == "get report":
        reports = os.listdir("Reports")
        if reports:
            reply = "Available reports:\n" + "\n".join(f"{i+1}. {name}" for i, name in enumerate(reports))
            reply += "\n\nType 'report <number>' to receive the report."
        else:
            reply = "No reports found."
        resp.message(reply)
    elif incoming_msg.startswith("report"):
        parts = incoming_msg.split()
        if len(parts) == 2 and parts[1].isdigit():
            index = int(parts[1]) - 1
            reports = os.listdir("Reports")
            if 0 <= index < len(reports):
                # Create a URL pointing to your report endpoint using your ngrok domain.
                file_url = f"(write your url)/report/{index}"
                msg = resp.message("Here is your report:")
                msg.media(file_url)
            else:
                resp.message("Invalid report number.")
        else:
            resp.message("Please use the format: report <number>")
    elif not incoming_msg:
        resp.message("Please enter a valid question.")
    else:
        try:
            reply_text = main_chain.invoke(incoming_msg)
        except Exception as e:
            reply_text = f"An error occurred: {str(e)}"
        resp.message(reply_text)
    return str(resp)

@app.route("/report/<int:report_id>", methods=["GET"])
def serve_report(report_id):
    reports = os.listdir("Reports")
    if 0 <= report_id < len(reports):
        return send_file(os.path.join("Reports", reports[report_id]), as_attachment=True)
    return "Report not found", 404

if __name__ == "__main__":
    app.run(debug=True)
