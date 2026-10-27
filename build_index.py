import chromadb
from sentence_transformers import SentenceTransformer

def chunk_text(text, chunk_size=80, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# 1. Read and chunk the document
with open("docs/faq.txt", encoding="utf-8") as f:
    chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]

# 2. Turn each chunk into an embedding
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks).tolist()

# 3. Store them in Chroma (saved in the chroma_db folder)
client = chromadb.PersistentClient(path="chroma_db")
try:
    client.delete_collection("faq")
except Exception:
    pass
collection = client.create_collection("faq")
collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings,
)
print("Stored", len(chunks), "chunks")

# 4. Test search: find the chunk closest in meaning to a question
questions = [
    "How much is the cheapest plan?",
    "What if my payment is late?",
    "My internet stopped working",
]
for q in questions:
    q_embedding = model.encode([q]).tolist()
    result = collection.query(query_embeddings=q_embedding, n_results=1)
    print()
    print("Question:", q)
    print("Best match:", result["documents"][0][0][:200])