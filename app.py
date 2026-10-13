import os
import gradio as gr
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import chromadb

DB_DIR = "./chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Read the network connection info injected into the container by Kubernetes
chroma_host = os.environ.get("CHROMA_SERVER_HOST", "chromadb-service")
chroma_port = os.environ.get("CHROMA_SERVER_HTTP_PORT", "8000")

# Connect directly to your live cluster database pod over the network!
remote_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
vector_db = Chroma(
    client=remote_client,
    embedding_function=embeddings
)

# Leave these lines commented out! Your CI/CD pipeline injects the key securely at runtime.
# PASTE YOUR NEW GROQ API KEY IN THE QUOTES BELOW and uncomment 
#os.environ["GROQ_API_KEY"]

# FIX: Swapped out the unrecognized string for an official Groq model ID
chat_model = ChatGroq(model="llama-3.1-8b-instant", 
                      temperature=0.2,
                      groq_api_key=os.environ.get("GROQ_API_KEY"))

def rag_interface(user_question):
    results = vector_db.similarity_search(user_question, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

    print("RAG Application Version 2.0 is now live!")

    messages = [
        ("system", "You are a precise assistant. Answer the user question strictly using the Context provided below. If the context does not contain the answer, reply that you don't know."),
        ("human", f"Context:\n{context_text}\n\nQuestion: {user_question}")
    ]
    try:
        response = chat_model.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error executing generation. System logs: {str(e)}"

demo = gr.Interface(
    fn=rag_interface, 
    inputs=gr.Textbox(label="Ask a question based on uploaded context:"), 
    outputs=gr.Textbox(label="Response:"),
    title="Free Offline-Compiled RAG App (Powered by Groq)"
)

if __name__ == "__main__":
    # Ensure server_name is set to "0.0.0.0" to listen across container boundaries
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False
    )
