'''Interface to ChromaDB'''

import itertools
from collections.abc import Iterable

import click
import chromadb
from tqdm import tqdm
from chromadb import QueryResult
from chromadb.config import Settings

from remem import chunker


_cached_setup = None


def setup(model_name: str, collection_name: str, db_path: str = 'chroma'):
    global _cached_setup
    if _cached_setup:
        return _cached_setup

    # Delay importing as it's slow
    import torch
    from sentence_transformers import SentenceTransformer

    def get_vram_gb() -> float:
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            return total / (1024 ** 3)
        return 0

    use_gpu = torch.cuda.is_available() and get_vram_gb() >= 3.5
    device = 'cuda' if use_gpu else 'cpu'

    click.echo(f'Loading embedding model on {device.upper()}...')

    model = SentenceTransformer(model_name, device=device)
    chroma_client = chromadb.PersistentClient(db_path, settings=Settings(anonymized_telemetry=False))
    collection = chroma_client.get_or_create_collection(collection_name)

    result = (model, collection)

    _cached_setup = result
    return result


def _get_setup():
    if not _cached_setup:
        raise RuntimeError(f'Call {setup.__name__} first.')
    return _cached_setup


def add(chunks: Iterable[chunker.Chunk], batch_size: int = 32) -> int:
    '''Add chunks into ChromaDB in batch'''
    model, collection = _get_setup()
    chunks = list(chunks)
    with tqdm(total=len(chunks)) as pbar:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings = model.encode(texts, normalize_embeddings=True)

            collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=[chunk.metadata for chunk in batch],
                ids=[chunk.id for chunk in batch],
            )
            pbar.update(len(batch))
    return len(chunks)


def query(keyword: str, instruction: str = '', n_results: int = 5) -> QueryResult:
    '''Query documents from ChromaDB with keyword'''
    model, collection = _get_setup()
    query_vec = model.encode(instruction + keyword, normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n_results,
    )
    return results


def update(chunks: Iterable[chunker.Chunk]) -> int:
    '''Add new chunks to ChromaDB, skipping existing ones'''
    chunks1, chunks2 = itertools.tee(chunks)
    _, collection = _get_setup()
    results = collection.get(ids=[c.id for c in chunks1])
    existing_ids = set(results['ids'])
    new_chunks = (c for c in chunks2 if c.id not in existing_ids)
    return add(new_chunks)
