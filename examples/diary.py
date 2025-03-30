#!/usr/bin/env python

import datetime
import dataclasses
from pprint import pprint
from dateutil.parser import isoparse
from collections.abc import Iterable

import click
from remem import chunker
from remem import chroma


# Example dataclass representing a diary post
@dataclasses.dataclass
class Post:
    id: int
    author: str
    title: str
    date: datetime.date
    content: str


# Generate 100 demo diary posts
def iter_post() -> Iterable[Post]:
    start_date = isoparse('2025-01-01').date()
    for i in range(100):
        yield Post(
            id=i,
            author='Alice',
            title=f'Title {i}',
            date=start_date + datetime.timedelta(days=i),
            content=f'This is post No. {i}, reflecting on day {i} of the year.',
        )


# Transform each Post into (metadata, text) tuple for ingestion.
def iter_ingestion_item() -> Iterable[chunker.IngestItem]:
    for post in iter_post():
        metadata = {
            'id': str(post.id),
            'author': post.author,
            'title': post.title,
            'date': post.date.isoformat(),
        }
        yield metadata, post.content


@click.group()
def cli():
    """A simple CLI for testing remem with diary-like posts."""
    # Initialize the embedding model and ChromaDB collection
    chroma.setup(model_name='all-MiniLM-L6-v2', collection_name='diary', db_path='alice_diary')


@cli.command()
def ingest():
    """Ingests sample diary posts into the vector database."""
    items = iter_ingestion_item()
    chunks = chunker.iter_chunk(items)
    chroma.add(chunks)
    click.echo('Ingested 100 posts into the vector DB.')


@cli.command()
@click.argument('keyword')
def query(keyword: str):
    """Query the vector database with a keyword."""
    results = chroma.query(keyword)
    pprint(results)


if __name__ == '__main__':
    cli()
