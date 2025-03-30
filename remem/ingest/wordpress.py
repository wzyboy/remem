#!/usr/bin/env python

"""
Ingester for WordPress posts via MySQL.

Connects to a WordPress MySQL database and retrieves published posts. For each post, it
extracts the post ID, date, title, and content, which can then be chunked and embedded
for later retrieval.
"""

import re
import html
import datetime
import dataclasses
from collections.abc import Iterable

import click
import pymysql
import pymysql.cursors
from remem import chunker


@dataclasses.dataclass
class Post:
    id: int
    dt: datetime.datetime
    title: str
    content: str

    def metadata(self) -> chunker.Metadata:
        return {
            'date': self.dt.date().isoformat(),
            'timestamp': self.dt.timestamp(),
            'title': self.title,
        }

    @classmethod
    def from_row(cls, row: dict):
        content = row['post_content'].replace('\r\n', '\n')
        content = re.sub(r'\n{2,}', '\n', content)
        content = html.unescape(content)
        return cls(
            id=row['ID'],
            dt=row['post_date'],
            title=row['post_title'],
            content=content,
        )

    def __str__(self) -> str:
        return f'{self.title}\n{self.content}'


def connect_mysql(user: str, password: str, database: str, host: str | None = None, unix_socket: str | None = None) -> pymysql.connections.Connection:
    """Establish a connection to the MySQL database via host or unix socket"""
    return pymysql.connect(
        user=user,
        password=password,
        host=host,
        database=database,
        unix_socket=unix_socket if unix_socket else None,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
    )


def iter_post(conn: pymysql.connections.Connection) -> Iterable[Post]:
    """Yield WordPress posts from the database"""
    query = """
        SELECT ID, post_date, post_title, post_content
        FROM wp_posts
        WHERE post_status = 'publish' AND post_type = 'post'
        ORDER BY ID ASC
    """

    with conn, conn.cursor() as cursor:
        cursor.execute(query)
        for row in cursor.fetchall():
            yield Post.from_row(row)


def iter_chunk(conn: pymysql.connections.Connection) -> Iterable[chunker.Chunk]:
    """Yield chunked WordPress posts"""
    items = ((post.metadata(), post.content) for post in iter_post(conn))
    yield from chunker.iter_chunk(items)


@click.group()
@click.option('--user', envvar='WP_USER', help='MySQL username (or set WP_USER)')
@click.option('--password', envvar='WP_PASSWORD', help='MySQL password (or set WP_PASSWORD)')
@click.option('--database', envvar='WP_DATABASE', default='wordpress', help='MySQL database name (or set WP_DATABASE)')
@click.option('--host', envvar='WP_HOST', default='localhost', help='MySQL host (or set WP_HOST)')
@click.option('--unix-socket', envvar='WP_SOCKET', type=click.Path(exists=True), help='MySQL Unix socket path (or set WP_SOCKET)')
@click.pass_context
def cli(ctx, host: str | None, unix_socket: str | None, user: str, password: str, database: str):
    ctx.ensure_object(dict)
    ctx.obj['conn'] = connect_mysql(user, password, database, host, unix_socket)


@cli.command()
@click.pass_obj
def preview(obj):
    """Preview chunked WordPress posts"""
    def iter_line():
        for chunk in iter_chunk(obj['conn']):
            yield str(chunk)
            yield '\n-----\n'

    click.echo_via_pager(iter_line())


if __name__ == '__main__':
    cli()
