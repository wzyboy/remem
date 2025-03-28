# remem

A simple tool to **chunk**, **embed**, and **recall** memory — from diaries to chat logs.

`remem` helps you structure, store, and retrieve text-based data (like journal entries or chat history) using embeddings and vector databases, for [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) workflow.

## ✨ Features

- 🔪 **Chunk** your input text into overlapping segments
- 📦 **Embed** using an embedding model
- 🧠 **Store** in a vector database (ChromaDB)
- 🔎 **Retrieve** contextually relevant chunks for LLM input

## Example

See [`diary.py`](./examples/diary.py) for how to use `remem` to ingest and query diary-like content.
