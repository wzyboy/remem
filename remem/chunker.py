'''Chunk text into pieces.'''

import hashlib
import dataclasses
from collections.abc import Iterable

import tiktoken


@dataclasses.dataclass
class Chunk:
    id: str
    metadata: dict
    text: str

    @classmethod
    def make(cls, metadata: dict, text: str):
        hash_data = f'{metadata}_{text}'
        hash = hashlib.sha1(hash_data.encode('utf-8')).hexdigest()
        return cls(hash, metadata, text)

    def __str__(self) -> str:
        token = len_token(self.text)
        return f'id={self.id} metadata={self.metadata!r} {token=}\n{self.text}'


IngestItem = tuple[dict[str, str], str]


def len_token(text: str, model: str = 'gpt-4o'):
    '''Return count of tokens in a given text'''
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))


seen_ids = set()


def iter_chunk(data: Iterable[IngestItem], max_len: int = 500, overlap: int = 1) -> Iterable[Chunk]:
    '''Take metadta + text and yield paragraph-aware chunks'''
    for md, text in data:
        # Split into paragraphs
        paragraphs = [p.rstrip() for p in text.split('\n') if p.strip()]
        buffer = []
        total_len = 0

        for para in paragraphs:
            if total_len + len_token(para) > max_len and buffer:
                chunk_text = '\n'.join(buffer)
                chunk = Chunk.make(metadata=md, text=chunk_text)
                if chunk.id not in seen_ids:
                    yield chunk
                seen_ids.add(chunk.id)

                # Overlap: keep last N paragraphs
                buffer = buffer[-overlap:] if overlap else []
                total_len = sum(len_token(p) for p in buffer)

            buffer.append(para)
            total_len += len_token(para)

        # Yield the final chunk
        if buffer:
            chunk_text = '\n'.join(buffer)
            chunk = Chunk.make(metadata=md, text=chunk_text)
            if chunk.id not in seen_ids:
                yield chunk
            seen_ids.add(chunk.id)
