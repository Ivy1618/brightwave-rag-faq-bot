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
    q_embedding = embedder.encode([question]).tolist()
    result = collection.query(query_embeddings=q_embedding, n_results=3)
    context = "\n\n---\n\n".join(result["documents"][0])
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
    text = response.choices[0].message.content
    return text.replace("\u202f", " ").replace("\u00a0", " ")

# (question, words that should appear in a correct answer; any one is enough)
answerable = [
    ("What time does support open?", ["8 am", "8:00"]),
    ("Is support open on Sundays?", ["closed", "not open", "no"]),
    ("How do I reset my router?", ["10 seconds"]),
    ("How much is the Premium plan?", ["799"]),
    ("What speed does the Premium plan have?", ["200"]),
    ("How can I pay my bill?", ["app"]),
    ("What is the late fee?", ["100"]),
    ("When might my service be suspended?", ["15 days"]),
    ("How do I report an outage?", ["app"]),
    ("What is the termination fee for cancelling early?", ["000"]),
    ("How much notice do I need to cancel?", ["30 days"]),
    ("Can I upgrade to Premium anytime?", ["any time", "anytime", "yes"]),
]
unanswerable = [
    "Can I pay with GCash?",
    "Do you offer cable TV packages?",
    "Do you have a 1 Gbps plan?",
]
unknown_words = ["not aware", "aware of", "not sure", "don't know", "do not                  know", "not listed",
                 "isn't listed", "no information", "doesn't mention",
                 "not mentioned", "don’t know", "isn’t listed"]

score = 0
for q, keywords in answerable:
    answer = ask(q)
    ok = any(k in answer.lower() for k in keywords)
    score += ok
    print("PASS" if ok else "FAIL", "|", q)
    if not ok:
        print("   Answer:", answer)

for q in unanswerable:
    answer = ask(q)
    ok = any(w in answer.lower() for w in unknown_words)
    score += ok
    print("PASS" if ok else "FAIL", "|", q, "(should say it doesn't know)")
    if not ok:
        print("   Answer:", answer)

print()
print("Score:", score, "out of", len(answerable) + len(unanswerable))