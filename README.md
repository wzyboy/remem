# remem

A simple tool to **chunk**, **embed**, and **recall** memory — from diaries to chat logs.

`remem` helps you structure, store, and retrieve text-based data (like journal entries or chat history) using embeddings and vector databases, for [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) workflow.

## ✨ Features

- 🔪 **Chunk** your input text into overlapping segments
- 📦 **Embed** using an embedding model
- 🧠 **Store** in a vector database (ChromaDB)
- 🔎 **Retrieve** contextually relevant chunks for LLM input

## Ingesters

### Telegram

See [remem.ingest.telegram](./remem/ingest/telegram.py). For Telegram chat history files produced by the [telegram-history-dump](https://github.com/tvdstaaij/telegram-history-dump).

### WordPress

See [remem.ingest.wordpress](./remem/ingest/wordpress.py). For WordPress posts in MySQL database.
