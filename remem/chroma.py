'''Interface to ChromaDB'''

import logging
import itertools
import dataclasses
from collections.abc import Iterable
from typing import TYPE_CHECKING

from tqdm import tqdm
from remem import chunker
from remem import utils

if TYPE_CHECKING:
    from chromadb import Collection
    from chromadb import QueryResult
    from chromadb.api import ClientAPI
    from sentence_transformers import SentenceTransformer


default_db_path = 'chroma'
log = logging.getLogger(__name__)


@dataclasses.dataclass
class ChromaSetup:
    model: 'SentenceTransformer'
    collection: 'Collection'
    retrieval_instruction: str


def get_client(db_path: str = default_db_path) -> 'ClientAPI':
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(db_path, settings=Settings(anonymized_telemetry=False))


def get_collection(collection_name: str, db_path: str = default_db_path) -> 'Collection':
    chroma_client = get_client(db_path)
    return chroma_client.get_or_create_collection(collection_name)


def get_setup(model_name: str, collection_name: str, retrieval_instruction: str = '', db_path: str = default_db_path) -> ChromaSetup:
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

    log.debug(f'Loading embedding model on {device.upper()}...')

    model = SentenceTransformer(model_name, device=device)
    collection = get_collection(collection_name, db_path)

    return ChromaSetup(model, collection, retrieval_instruction)


def add(cs: ChromaSetup, chunks: Iterable[chunker.Chunk], batch_size: int = 32) -> int:
    '''Add chunks into ChromaDB with progress bar'''
    chunks = list(chunks)
    with tqdm(total=len(chunks)) as pbar:
        for batch in utils.batched(chunks, batch_size):
            texts = [chunk.text for chunk in batch]
            embeddings = cs.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

            cs.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=[chunk.metadata for chunk in batch],
                ids=[chunk.id for chunk in batch],
            )
            pbar.update(len(batch))
    return len(chunks)


def query(cs: ChromaSetup, keyword: str, n_results: int = 5) -> 'QueryResult':
    '''Query documents from ChromaDB with keyword'''
    query_vec = cs.model.encode(cs.retrieval_instruction + keyword, normalize_embeddings=True, show_progress_bar=False).tolist()
    results = cs.collection.query(
        query_embeddings=[query_vec],
        n_results=n_results,
    )
    return results


def update(cs: ChromaSetup, chunks: Iterable[chunker.Chunk]) -> int:
    '''Add new chunks to ChromaDB, skipping existing ones'''
    chunks1, chunks2 = itertools.tee(chunks)
    del chunks
    # We cannot retrieve too many IDs at the same time or we get
    # "sqlite3.OperationalError: too many SQL variables" exception.
    existing_ids = set()
    for chunk_batch in utils.batched(chunks1, 128):
        ids = [c.id for c in chunk_batch]
        results = cs.collection.get(ids=ids)
        existing_ids.update(results['ids'])

    # Add new chunks
    new_chunks = (c for c in chunks2 if c.id not in existing_ids)
    return add(cs, new_chunks)
