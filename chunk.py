def chunk_text(text, chunk_size=80, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

with open("docs/faq.txt", encoding="utf-8") as f:
    text = f.read()

chunks = chunk_text(text)

print("Number of chunks:", len(chunks))
print()
for i, chunk in enumerate(chunks[:2]):
    print("Chunk", i + 1, ":")
    print(chunk)
    print()