import os
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

from gesda_RAG import RAG

# --- Load model and index once at startup ---
INDEX_PATH = "/mnt/nlp/scratch/home/belghmi/indexes/gesda_bge-m3"
EMBEDDING_MODEL_PATH = "/mnt/nlp/scratch/home/belghmi/models/bge-m3"
LLM_MODEL_NAME = "swiss-ai/Apertus-70B-Instruct-2509"

print("Initializing GESDA RAG pipeline...")
rag = RAG(
    index_path=INDEX_PATH,
    embedding_model_path=EMBEDDING_MODEL_PATH,
    llm_model_name=LLM_MODEL_NAME,
    retrieval_k=5,
)
print("Ready.")


def answer(query: str) -> str:
    if not query.strip():
        return "Please enter a question."
    docs = rag.get_most_relevant_docs(query)
    return rag.generate_answer(query, docs)


demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask a question about the GESDA Science Breakthrough Radar...",
        label="Your question",
    ),
    outputs=gr.Textbox(label="Answer", lines=15),
    title="GESDA Science Breakthrough Radar — Q&A",
    description="Ask anything about science and technology foresight from the GESDA radar reports (2021–2026). Multilingual.",
)

if __name__ == "__main__":
    demo.launch(share=True)
