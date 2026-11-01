from textwrap import dedent

import pytest

from reporag.ingest import chunking
from reporag.ingest.chunking import chunk_code, chunk_file, chunk_markdown, window_chunks


def _lines_of(source: str, chunk) -> str:
    return "\n".join(source.splitlines()[chunk.start_line - 1 : chunk.end_line])


PY_SOURCE = dedent(
    '''\
    """Module docstring."""
    import os

    CONSTANT = 3


    def add(a, b):
        return a + b


    @decorator
    def decorated():
        pass


    class Greeter:
        """Says hello."""

        def hello(self, name):
            return f"hi {name}"
    '''
)


def test_python_splits_at_functions_and_classes():
    chunks = chunk_code("pkg/mod.py", "python", PY_SOURCE)
    by_symbol = {c.symbol: c for c in chunks}

    assert by_symbol["add"].kind == "function"
    assert by_symbol["add"].text.startswith("def add")
    assert by_symbol["decorated"].text.startswith("@decorator")  # decorator kept
    assert by_symbol["Greeter"].kind == "class"
    module = [c for c in chunks if c.kind == "module"]
    assert module and "import os" in module[0].text


def test_line_numbers_match_source():
    for chunk in chunk_code("pkg/mod.py", "python", PY_SOURCE):
        assert chunk.text == _lines_of(PY_SOURCE, chunk)
        assert 1 <= chunk.start_line <= chunk.end_line
        assert chunk.path == "pkg/mod.py" and chunk.language == "python"


def test_large_class_is_split_into_methods(monkeypatch):
    monkeypatch.setattr(chunking, "MAX_CHUNK_LINES", 6)
    source = dedent(
        """\
        class Big:
            x = 1

            def one(self):
                a = 1
                return a

            # explains two
            def two(self):
                return 2
        """
    )
    chunks = chunk_code("big.py", "python", source)
    symbols = [c.symbol for c in chunks]
    assert "Big.one" in symbols and "Big.two" in symbols
    two = next(c for c in chunks if c.symbol == "Big.two")
    assert "# explains two" in two.text  # leading comment travels with the method
    assert all(c.text == _lines_of(source, c) for c in chunks)


def test_javascript_functions_arrows_and_jsdoc():
    source = dedent(
        """\
        const lib = require('lib');

        /**
         * Adds numbers.
         */
        function add(a, b) {
          return a + b;
        }

        export const double = (x) => x * 2;

        class Stack {
          push(v) { this.items.push(v); }
        }
        """
    )
    chunks = chunk_code("src/util.js", "javascript", source)
    by_symbol = {c.symbol: c for c in chunks}
    assert by_symbol["add"].text.startswith("/**")  # JSDoc attached
    assert by_symbol["double"].kind == "function"
    assert by_symbol["Stack"].kind == "class"
    assert any("require('lib')" in c.text for c in chunks if c.kind == "module")


def test_typescript_interfaces_and_types():
    source = dedent(
        """\
        export interface Options {
          retries: number;
        }

        export type Method = 'get' | 'post';

        export function request(opts: Options): void {}
        """
    )
    chunks = chunk_code("src/types.ts", "typescript", source)
    kinds = {c.symbol: c.kind for c in chunks}
    assert kinds == {"Options": "type", "Method": "type", "request": "function"}


def test_markdown_splits_by_heading_and_ignores_code_fences():
    source = dedent(
        """\
        Intro text.

        # Install
        Run pip.

        ```bash
        # not a heading
        pip install x
        ```

        ## macOS
        Use brew.

        # Usage
        Call it.
        """
    )
    chunks = chunk_markdown("README.md", source)
    symbols = [c.symbol for c in chunks]
    assert symbols == [None, "Install", "Install > macOS", "Usage"]
    assert "# not a heading" in chunks[1].text


def test_window_chunks_overlap_and_cover_everything():
    lines = [f"line {i}" for i in range(1, 26)]
    chunks = window_chunks("a.txt", "text", lines, size=10, overlap=3)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 10), (8, 17), (15, 24), (22, 25)]


def test_window_rejects_bad_overlap():
    with pytest.raises(ValueError):
        window_chunks("a.txt", "text", ["x"], size=5, overlap=5)


def test_chunk_file_falls_back_to_windows_for_unknown_language():
    chunks = chunk_file("main.go", "go", "package main\n\nfunc main() {}\n")
    assert len(chunks) == 1 and chunks[0].kind == "window"


def test_blank_file_produces_no_chunks():
    assert chunk_file("empty.py", "python", "") == []
    assert chunk_file("blank.md", "markdown", "\n\n") == []
