import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")
collection = chromadb.PersistentClient(path="chroma_db").get_collection("faq")

def ask(question):
    # 1. Retrieve the 3 most relevant chunks
    q_embedding = embedder.encode([question]).tolist()
    result = collection.query(query_embeddings=q_embedding, n_results=3)
    chunks = result["documents"][0]
    context = "\n\n---\n\n".join(chunks)

    # 2. Ask the model to answer using only those chunks
    prompt = (
        "You are a support assistant for Brightwave Internet Services. "
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know and "
        "suggest contacting support. Keep the answer short.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content

    print("\nQuestion:", question)
    print("Answer:", answer)
    print("Sources:", [c.split("\n")[0] for c in chunks])

questions = [
    "How much is the cheapest plan?",
    "My internet stopped working",
    "Can I pay with GCash?",
    "Do you offer cable TV packages?",
]
for q in questions:
    ask(q)