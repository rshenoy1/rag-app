import os
import gradio as gr
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

DB_DIR = "./chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(DB_DIR):
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
    raise FileNotFoundError("Chroma database directory missing. Run ingest.py first.")

# PASTE YOUR NEW GROQ API KEY IN THE QUOTES BELOW and uncomment 
#os.environ["GROQ_API_KEY"] = "TODO: PASTE YOUR GROQ API KEY HERE"

# Using Llama 3.1 8B on Groq for ultra-fast, free conversational intelligence
chat_model = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.2)

def rag_interface(user_question):
    results = vector_db.similarity_search(user_question, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    
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
    demo.launch(share=True)
