#!/usr/bin/env python

'''
Ingester for Telegram history dump.

This script processes Telegram chat export files in `.jsonl` format, as
produced by the `telegram-history-dump` tool:

https://github.com/tvdstaaij/telegram-history-dump

It recursively discovers all `.jsonl` files within the specified files or
directories. For each file, it reads message events and groups them into "chat
sessions" based on a configurable time gap (default: 2 hours). A new session is
created whenever the time between two consecutive messages exceeds this
threshold.

Each session contains:
- A name derived from the participants or group title
- A start and end datetime
- The full text content of the session, including media annotations
'''

import json
import datetime
import dataclasses
from pathlib import Path
from collections.abc import Iterable

import click
from remem import chunker
from remem import utils


@dataclasses.dataclass(order=True)
class ChatMessage:
    id: str
    reply_id: str | None
    reply_text: str | None
    dt: datetime.datetime
    from_name: str
    to_name: str
    group_name: str | None
    text: str

    @classmethod
    def from_dict(cls, event: dict):
        assert event.get('event') == 'message', 'Not a message'
        assert event.get('text') or event.get('media'), 'No text/media in message'
        return cls(
            id=event['id'],
            reply_id=event.get('reply_id'),
            reply_text=None,  # this is populated with ChatSession
            dt=datetime.datetime.fromtimestamp(event['date'], tz=datetime.UTC),
            from_name=cls._extract_name(event['from']),
            to_name=cls._extract_name(event['to']),
            group_name=event['to'].get('title'),
            text=cls._extract_text(event),
        )

    def __str__(self) -> str:
        return f'{self.from_name}: {self.text}'

    @staticmethod
    def _extract_name(peer: dict) -> str:
        peer_type = peer['peer_type']
        if peer_type == 'user':
            first = peer.get('first_name', '')
            last = peer.get('last_name', '')
            return (first + ' ' + last).strip() or f'{peer_type}#{peer["peer_id"]}'
        elif peer_type in ('chat', 'channel'):
            return peer['title']
        else:
            raise ValueError(f'Invalid {peer_type=}')

    @staticmethod
    def _extract_text(event: dict) -> str:
        text = event.get('text', '')
        media = event.get('media')
        mtext = ''

        if media:
            mtype = media.get('type')

            match mtype:
                case 'photo':
                    mtext = '[PHOTO]'
                    caption = str(media.get('caption', '')).strip()
                    if caption:
                        mtext += f' {caption}'

                case 'webpage':
                    parts = [
                        chunker.truncate_by_tokens(str(media.get(k, '')), 20)
                        for k in ('title', 'description', 'author')
                    ]
                    parts = [p for p in parts if p]
                    mtext = '[WEBPAGE] ' + ' - '.join(parts) if parts else '[WEBPAGE]'

                case 'document':
                    mtext = '[DOCUMENT]'

                case 'video':
                    mtext = '[VIDEO]'

                case 'audio':
                    mtext = '[AUDIO]'

                case 'geo':
                    lat = media.get('latitude')
                    lon = media.get('longitude')
                    if lat is not None and lon is not None:
                        mtext = f'[LOCATION] ({lat}, {lon})'
                    else:
                        mtext = '[LOCATION]'

                case 'unsupported':
                    mtext = '[UNSUPPORTED MEDIA]'

                case _:
                    mtext = f'[UNKNOWN MEDIA TYPE: {mtype}]'

        # Combine text and media placeholder
        return f'{text} {mtext}' if text and mtext else (mtext or text)


@dataclasses.dataclass
class ChatSession:
    name: str
    dt_start: datetime.datetime
    dt_end: datetime.datetime
    content: str

    @classmethod
    def from_messages(cls, _messages: Iterable[ChatMessage]):
        # NOTE: this list might be modified in-place.
        messages = sorted(_messages)
        assert messages, 'No messages'
        msg_id_map = {msg.id: msg for msg in messages}

        # The messages are from the same file, so they are either all private
        # messages or all group messages. For private chats, use names of both
        # peers as the session name. For group chats, use the group title as
        # session name.
        _peer_names = set()
        _group_name = None
        for msg in messages:
            cls._attach_reply(msg, msg_id_map)
            if not _group_name:
                _peer_names.add(msg.from_name)
                _peer_names.add(msg.to_name)
                if msg.group_name:
                    _group_name = msg.group_name

        name = ' & '.join(sorted(_peer_names)) if not _group_name else _group_name

        # Group consecutive messages by speaker with indentation
        lines = []
        last_speaker = None

        for msg in messages:
            if msg.reply_text:
                quote = chunker.truncate_by_tokens(msg.reply_text, 10)
                text = f'↩[{quote}] {msg.text}'
            else:
                text = msg.text

            if msg.from_name != last_speaker:
                lines.append(f'{msg.from_name}:')
                last_speaker = msg.from_name

            lines.extend(f'\t{line.strip()}' for line in text.splitlines() if line.strip())

        content = '\n'.join(lines)

        return cls(
            name=name,
            dt_start=messages[0].dt,
            dt_end=messages[-1].dt,
            content=content,
        )

    @staticmethod
    def _attach_reply(current_msg: ChatMessage, msg_id_map: dict[str, ChatMessage]):
        if reply_id := current_msg.reply_id:
            if reply_msg := msg_id_map.get(reply_id):
                current_msg.reply_text = reply_msg.text

    def metadata(self) -> chunker.Metadata:
        return {
            'name': self.name,
            'date': self.dt_start.date().isoformat(),
            'timestamp': self.dt_start.timestamp(),
        }

    def __str__(self) -> str:
        return f'{self.dt_start.date().isoformat()} | {self.name}\n{self.content}'


def iter_chat_session(jsonl: Path, time_gap: datetime.timedelta = datetime.timedelta(hours=2)):
    '''Read .jsonl file and group messages into chat sessions'''
    with open(jsonl, encoding='utf-8') as f:
        dt_cursor = None
        buffer = []
        for line in f:
            event = json.loads(line)
            if event.get('event') != 'message':
                continue
            if not event.get('text') and not event.get('media'):
                continue
            msg = ChatMessage.from_dict(event)
            # If time between two messages > gap, build a session with messages in buffer
            if dt_cursor is not None and abs(dt_cursor - msg.dt) >= time_gap:
                yield ChatSession.from_messages(buffer)
                buffer.clear()
            # Otherwise, add message to buffer
            buffer.append(msg)
            dt_cursor = msg.dt
        # Yield the last session
        if buffer:
            yield ChatSession.from_messages(buffer)


def iter_chunk(jsonl: Path) -> Iterable[chunker.Chunk]:
    '''Read .jsonl file and return chunked chat sessions'''
    items = (
        (s.metadata(), s.content)
        for s in iter_chat_session(jsonl)
    )
    yield from chunker.iter_chunk(items, overlap=5)


@click.group()
def cli():
    pass


@cli.command()
@click.argument('path', type=click.Path(path_type=Path, exists=True))
def preview_chats(path: Iterable[Path]):
    '''Preview chat sessions from file or directory'''
    def iter_line():
        for file in utils.iter_files(path, '.jsonl'):
            for session in iter_chat_session(file):
                yield str(session)
                yield '\n-----\n'
    click.echo_via_pager(iter_line())


@cli.command()
@click.argument('path', type=click.Path(path_type=Path, exists=True))
def preview_chunks(path: Iterable[Path]):
    '''Preview chunked chat sessions from file or directory'''
    def iter_line():
        for file in utils.iter_files(path, '.jsonl'):
            for chunk in iter_chunk(file):
                yield str(chunk)
                yield '\n-----\n'

    click.echo_via_pager(iter_line())


if __name__ == '__main__':
    cli()
