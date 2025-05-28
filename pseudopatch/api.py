from __future__ import annotations

import pathlib
from typing import Callable, Dict, List, Tuple

from .domain import Commit, Patch
from .exceptions import DiffError
from .patching import Parser, patch_to_commit


def text_to_patch(text: str, orig: Dict[str, str]) -> Tuple[Patch, int]:
    lines = text.splitlines()
    if (
        len(lines) < 2
        or not Parser._norm(lines[0]).startswith("*** Begin Patch")
        or Parser._norm(lines[-1]) != "*** End Patch"
    ):
        raise DiffError("Invalid patch text - missing sentinels")

    parser = Parser(current_files=orig, lines=lines, index=1)
    parser.parse()
    return parser.patch, parser.fuzz


def identify_files_needed(text: str) -> List[str]:
    lines = text.splitlines()
    return [
        line[len("*** Update File: "):]
        for line in lines
        if line.startswith("*** Update File: ")
    ] + [
        line[len("*** Delete File: "):]
        for line in lines
        if line.startswith("*** Delete File: ")
    ]


def identify_files_added(text: str) -> List[str]:
    lines = text.splitlines()
    return [
        line[len("*** Add File: "):]
        for line in lines
        if line.startswith("*** Add File: ")
    ]


def load_files(paths: List[str], open_fn: Callable[[str], str]) -> Dict[str, str]:
    return {path: open_fn(path) for path in paths}


def apply_commit(
    commit: Commit,
    write_fn: Callable[[str, str], None],
    remove_fn: Callable[[str], None],
) -> None:
    for path, change in commit.changes.items():
        if change.type == "delete":
            remove_fn(path)
        elif change.type == "add":
            if change.new_content is None:
                raise DiffError(f"ADD change for {path} has no content")
            write_fn(path, change.new_content)
        elif change.type == "update":
            if change.new_content is None:
                raise DiffError(f"UPDATE change for {path} has no new content")
            target = change.move_path or path
            write_fn(target, change.new_content)
            if change.move_path:
                remove_fn(path)


def process_patch(
    text: str,
    open_fn: Callable[[str], str],
    write_fn: Callable[[str, str], None],
    remove_fn: Callable[[str], None],
) -> str:
    if not text.startswith("*** Begin Patch"):
        raise DiffError("Patch text must start with *** Begin Patch")
    paths = identify_files_needed(text)
    orig = load_files(paths, open_fn)
    patch, _fuzz = text_to_patch(text, orig)
    commit = patch_to_commit(patch, orig)
    apply_commit(commit, write_fn, remove_fn)
    return "Done!"
