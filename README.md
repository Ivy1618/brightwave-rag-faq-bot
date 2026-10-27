# Brightwave FAQ Bot (RAG)

A question-answering bot that answers customer questions using only a company FAQ. Built in Python as a learning project.

## How it works
1. The FAQ is split into one chunk per question.
2. Each chunk is turned into an embedding (sentence-transformers, all-MiniLM-L6-v2) and stored in ChromaDB.
3. For each question, the 3 closest chunks are retrieved.
4. The Groq API (openai/gpt-oss-20b) writes an answer using only those chunks, and says it doesn't know if the answer isn't there.

## Results
Tested on 15 questions (12 answerable, 3 not in the FAQ): 15 out of 15 passed. My first run scored 11/15, but most of those failures were my keyword check being too strict (special characters and different phrasing), not wrong answers. I fixed the checker, not the bot.

## What I learned
- Chunking by word count cut answers in half, and a vague question retrieved the wrong section. Splitting by question fixed this.
- Retrieving 3 chunks instead of 1 let the model combine related answers.
- My first keyword-based test gave false failures (special characters, different phrasing), so automatic checks need a manual look.
- Groq retired the model I first used (llama-3.1-8b-instant), so I switched to the recommended replacement.

## Limitations
Small FAQ (9 Q&As), a test set I wrote myself, and a simple keyword check. Not production-ready.

## Run it
pip install sentence-transformers chromadb groq python-dotenv
Add GROQ_API_KEY to a .env file, then run build_index.py and rag.py.
