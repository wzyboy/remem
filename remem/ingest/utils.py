import itertools
from pathlib import Path
from collections.abc import Iterable


def iter_files(paths: Path | Iterable[Path], suffix: str) -> Iterable[Path]:
    '''Find all files with the given suffix in paths recursively'''
    if isinstance(paths, Path):
        paths = [paths]
    def gen():
        for path in paths:
            if path.is_dir():
                yield (p for p in path.rglob(f'*{suffix}') if p.is_file())
            elif path.suffix == suffix:
                yield [path]
    yield from itertools.chain.from_iterable(gen())
