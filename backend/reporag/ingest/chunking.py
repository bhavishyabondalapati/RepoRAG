"""Code-aware chunking.

- Python / JavaScript / TypeScript: parsed with tree-sitter and split at
  function/class boundaries. Large classes are split into their methods.
- Markdown: split by headings, keeping the heading breadcrumb as metadata.
- Everything else: fixed-size line windows with overlap.

Every chunk keeps its file path, language, and 1-based inclusive line range so
answers can cite exact locations.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache

MAX_CHUNK_LINES = 120  # definitions longer than this are split further
WINDOW_LINES = 60
WINDOW_OVERLAP = 10


@dataclass
class Chunk:
    path: str
    language: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    text: str
    kind: str  # function | class | method | type | module | section | window
    symbol: str | None = None

    @property
    def id(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def search_text(self) -> str:
        """Text used for embedding and BM25: adds the path and symbol as context."""
        header = f"File: {self.path}"
        if self.symbol:
            header += f"\nSymbol: {self.symbol}"
        return f"{header}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Tree-sitter setup
# --------------------------------------------------------------------------- #

TREE_SITTER_LANGUAGES = {"python", "javascript", "typescript", "tsx"}


@lru_cache(maxsize=None)
def _parser(language: str):
    from tree_sitter import Language, Parser

    if language == "python":
        import tree_sitter_python as mod

        lang = Language(mod.language())
    elif language == "javascript":
        import tree_sitter_javascript as mod

        lang = Language(mod.language())
    elif language == "typescript":
        import tree_sitter_typescript as mod

        lang = Language(mod.language_typescript())
    elif language == "tsx":
        import tree_sitter_typescript as mod

        lang = Language(mod.language_tsx())
    else:
        raise ValueError(f"No tree-sitter grammar for {language}")
    return Parser(lang)


_FUNCTION_TYPES = {
    "function_definition",  # python
    "function_declaration",
    "generator_function_declaration",
}
_CLASS_TYPES = {"class_definition", "class_declaration", "abstract_class_declaration"}
_TYPE_TYPES = {"interface_declaration", "type_alias_declaration", "enum_declaration"}
_VAR_DECL_TYPES = {"lexical_declaration", "variable_declaration"}
_FUNCTION_VALUE_TYPES = {"arrow_function", "function_expression", "function", "class"}
_METHOD_TYPES = {"function_definition", "decorated_definition", "method_definition"}
_COMMENT_TYPES = {"comment"}


def _unwrap(node):
    """Look through wrappers (decorators, `export`) to the real definition node."""
    inner = node
    while inner.type in {"decorated_definition", "export_statement"}:
        target = inner.child_by_field_name("definition") or inner.child_by_field_name(
            "declaration"
        )
        if target is None:
            break
        inner = target
    return inner


def _name_of(node) -> str | None:
    name = node.child_by_field_name("name")
    if name is not None:
        return name.text.decode("utf8", "replace")
    if node.type in _VAR_DECL_TYPES:
        for child in node.named_children:
            if child.type == "variable_declarator":
                n = child.child_by_field_name("name")
                return n.text.decode("utf8", "replace") if n is not None else None
    return None


def _definition_kind(node) -> str | None:
    """Return 'function' | 'class' | 'type' if this top-level node is a definition."""
    inner = _unwrap(node)
    if inner.type in _FUNCTION_TYPES:
        return "function"
    if inner.type in _CLASS_TYPES:
        return "class"
    if inner.type in _TYPE_TYPES:
        return "type"
    if inner.type in _VAR_DECL_TYPES:
        for child in inner.named_children:
            if child.type == "variable_declarator":
                value = child.child_by_field_name("value")
                if value is not None and value.type in _FUNCTION_VALUE_TYPES:
                    return "function"
    return None


def _rows(node) -> tuple[int, int]:
    """0-based inclusive start/end rows for a node."""
    start, end = node.start_point[0], node.end_point[0]
    if node.end_point[1] == 0 and end > start:
        end -= 1
    return start, end


# --------------------------------------------------------------------------- #
# Chunk builders
# --------------------------------------------------------------------------- #


def _make(path, language, lines, start, end, kind, symbol=None) -> Chunk | None:
    """Build a chunk from 0-based inclusive rows; returns None for blank spans."""
    text = "\n".join(lines[start : end + 1])
    if not text.strip():
        return None
    return Chunk(path, language, start + 1, end + 1, text, kind, symbol)


def window_chunks(
    path: str,
    language: str,
    lines: list[str],
    start: int = 0,
    end: int | None = None,
    kind: str = "window",
    symbol: str | None = None,
    size: int = WINDOW_LINES,
    overlap: int = WINDOW_OVERLAP,
) -> list[Chunk]:
    """Split rows [start, end] (0-based inclusive) into overlapping windows."""
    if end is None:
        end = len(lines) - 1
    if size <= overlap:
        raise ValueError("window size must be larger than overlap")
    chunks = []
    pos = start
    while pos <= end:
        stop = min(pos + size - 1, end)
        chunk = _make(path, language, lines, pos, stop, kind, symbol)
        if chunk:
            chunks.append(chunk)
        if stop == end:
            break
        pos = stop + 1 - overlap
    return chunks


def _sized(path, language, lines, start, end, kind, symbol) -> list[Chunk]:
    """One chunk if it fits, otherwise overlapping windows tagged with the symbol."""
    if end - start + 1 <= MAX_CHUNK_LINES:
        chunk = _make(path, language, lines, start, end, kind, symbol)
        return [chunk] if chunk else []
    return window_chunks(path, language, lines, start, end, kind=kind, symbol=symbol)


def _class_members(class_node):
    body = class_node.child_by_field_name("body")
    if body is None:
        return []
    return [c for c in body.named_children if c.type in _METHOD_TYPES]


def _split_class(path, language, lines, node, start, end, class_name) -> list[Chunk]:
    """Split an oversized class into a header chunk plus one chunk per method."""
    members = _class_members(_unwrap(node))
    if not members:
        return _sized(path, language, lines, start, end, "class", class_name)

    chunks: list[Chunk] = []
    first_member_start = _rows(members[0])[0]
    header_end = max(start, first_member_start - 1)
    chunks += _sized(path, language, lines, start, header_end, "class", class_name)

    cursor = first_member_start
    for member in members:
        m_start, m_end = _rows(member)
        # Pull in comments sitting between the previous method and this one.
        if cursor < m_start and _is_blank_or_comment(lines, cursor, m_start - 1):
            m_start = cursor
        method_name = _name_of(_unwrap(member)) or "<anonymous>"
        chunks += _sized(
            path, language, lines, m_start, m_end, "method", f"{class_name}.{method_name}"
        )
        cursor = m_end + 1
    if cursor <= end:  # trailing class content (e.g. closing brace + fields)
        tail = _make(path, language, lines, cursor, end, "class", class_name)
        if tail and len(tail.text.strip()) > 2:
            chunks.append(tail)
    return chunks


def _is_blank_or_comment(lines, start, end) -> bool:
    for line in lines[start : end + 1]:
        s = line.strip()
        if s and not s.startswith(("#", "//", "/*", "*", "*/")):
            return False
    return True


def chunk_code(path: str, language: str, source: str) -> list[Chunk]:
    """Split code at top-level definitions using tree-sitter."""
    lines = source.splitlines()
    if not lines:
        return []
    try:
        tree = _parser(language).parse(source.encode("utf8"))
    except Exception:
        return window_chunks(path, language, lines)

    chunks: list[Chunk] = []
    pending_start: int | None = None  # start row of accumulated module-level code
    pending_end = -1

    def flush_pending(until_row: int) -> None:
        nonlocal pending_start
        if pending_start is not None and pending_start <= until_row:
            chunks.extend(
                _sized(path, language, lines, pending_start, until_row, "module", None)
            )
        pending_start = None

    nodes = tree.root_node.children
    for i, node in enumerate(nodes):
        start, end = _rows(node)
        kind = _definition_kind(node)

        if kind is None:
            # Comments directly above a definition belong to it (docstrings / JSDoc).
            nxt = nodes[i + 1] if i + 1 < len(nodes) else None
            if (
                node.type in _COMMENT_TYPES
                and nxt is not None
                and _rows(nxt)[0] - end <= 1
                and _definition_kind(nxt) is not None
            ):
                continue
            if pending_start is None:
                pending_start = start
            pending_end = end
            # Keep module chunks bounded.
            if pending_end - pending_start + 1 >= MAX_CHUNK_LINES:
                flush_pending(pending_end)
            continue

        # Attach leading comment block (skipped above) to this definition.
        def_start = start
        j = i - 1
        while j >= 0 and nodes[j].type in _COMMENT_TYPES and def_start - _rows(nodes[j])[1] <= 1:
            def_start = _rows(nodes[j])[0]
            j -= 1

        flush_pending(min(pending_end, def_start - 1))
        name = _name_of(_unwrap(node))
        if kind == "class" and end - def_start + 1 > MAX_CHUNK_LINES:
            chunks += _split_class(path, language, lines, node, def_start, end, name)
        else:
            chunks += _sized(path, language, lines, def_start, end, kind, name)

    flush_pending(pending_end)
    return chunks


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def chunk_markdown(path: str, source: str) -> list[Chunk]:
    """Split markdown by ATX headings (ignoring '#' inside fenced code blocks)."""
    lines = source.splitlines()
    sections: list[tuple[int, list[str]]] = []  # (start_row, breadcrumb)
    stack: list[tuple[int, str]] = []  # (level, title)
    in_fence = False
    for row, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append((row, [t for _, t in stack]))

    if not sections or sections[0][0] > 0:
        sections.insert(0, (0, []))

    chunks: list[Chunk] = []
    for idx, (start, crumbs) in enumerate(sections):
        end = sections[idx + 1][0] - 1 if idx + 1 < len(sections) else len(lines) - 1
        symbol = " > ".join(crumbs) if crumbs else None
        chunks += _sized(path, "markdown", lines, start, end, "section", symbol)
    return chunks


def chunk_file(path: str, language: str, source: str) -> list[Chunk]:
    """Dispatch to the right chunker for a file's language."""
    if language in TREE_SITTER_LANGUAGES:
        return chunk_code(path, language, source)
    if language == "markdown":
        return chunk_markdown(path, source)
    return window_chunks(path, language, source.splitlines())
