"""Exact lexical links, source spans and their undirected paragraph projection."""

import re
import unicodedata as ud
from collections import defaultdict
from collections.abc import Iterator
from itertools import combinations
from typing import TypedDict

from .documents import STOPWORDS, Documents, digest, validate_documents


class Paragraph(TypedDict):
    id: str
    title: str
    text: str


class Entity(TypedDict):
    id: str
    title: str
    aliases: list[str]


class Owns(TypedDict):
    id: str
    paragraph_id: str
    entity_id: str


class Mention(Owns):
    text: str
    start: int
    end: int
    alias: str


class Link(TypedDict):
    source: str
    target: str
    evidence_edge_ids: list[str]


class Graph(TypedDict):
    paragraphs: list[Paragraph]
    entities: list[Entity]
    owns: list[Owns]
    mentions: list[Mention]
    projection: list[Link]


def normalize(text: str) -> str:
    return " ".join(ud.normalize("NFKC", text).casefold().split())


WORD_CHARACTER = re.compile(r"\w")


type Origin = tuple[str, int, int]


def _decomposed(text: str) -> Iterator[Origin]:
    # Canonically reorder each combining run without losing its source indices.
    marks: list[Origin] = []
    for i, char in enumerate(text):
        for unit in ud.normalize("NFKD", char):
            token = (unit, i, i + 1)
            if ud.combining(unit):
                marks.append(token)
            else:
                yield from sorted(marks, key=lambda t: ud.combining(t[0]))
                marks.clear()
                yield token
    yield from sorted(marks, key=lambda t: ud.combining(t[0]))


def _nfkc_origins(text: str) -> Iterator[Origin]:
    if ud.is_normalized("NFKC", text):
        yield from ((char, i, i + 1) for i, char in enumerate(text))
        return
    # Canonical composition can join nonadjacent marks or consecutive Hangul jamo.
    segment: list[Origin] = []
    last_class = 0
    for char, start, end in _decomposed(text):
        combining = ud.combining(char)
        if segment and (last_class == 0 or last_class < combining):
            starter, left, right = segment[0]
            composed = ud.normalize("NFC", starter + char)
            if len(composed) == 1:
                segment[0] = (composed, min(left, start), max(right, end))
                continue
        if combining == 0:
            yield from segment
            segment = []
        segment.append((char, start, end))
        last_class = combining
    yield from segment


def _search_text(text: str) -> tuple[str, list[tuple[int, int]]]:
    chars: list[str] = []
    spans: list[tuple[int, int]] = []
    for char, start, end in _nfkc_origins(text):
        for folded in char.casefold():
            if folded.isspace():
                if chars and chars[-1] == " ":
                    spans[-1] = (spans[-1][0], end)
                    continue
                folded = " "
            chars.append(folded)
            spans.append((start, end))
    return "".join(chars), spans


def _short_title(name: str) -> str:
    if name.endswith(")"):
        depth = 0
        for i in range(len(name) - 1, -1, -1):
            if name[i] == ")":
                depth += 1
            elif name[i] == "(":
                depth -= 1
                if depth == 0:
                    return name[:i].rstrip()
    return name


def _entities(titles: dict[str, set[str]]) -> list[Entity]:
    candidates: dict[str, set[str]] = defaultdict(set)
    for name in titles:
        # Only a trailing parenthetical disambiguator creates a shorter alias.
        alias = _short_title(name)
        if alias != name and sum(c.isalpha() for c in alias) >= 3 and alias not in STOPWORDS:
            candidates[alias].add(name)
    entities: list[Entity] = []
    for name in sorted(titles):
        aliases = [name]
        for alias, owners in candidates.items():
            if owners == {name} and alias not in titles:
                aliases.append(alias)
        entities.append(
            {"id": "e:" + digest([name]), "title": min(titles[name]), "aliases": sorted(aliases)}
        )
    return entities


def build_graph(documents: Documents) -> Graph:
    docs = validate_documents(documents)
    titles: dict[str, set[str]] = defaultdict(set)
    for doc in docs:
        titles[normalize(doc.title)].add(doc.title)
    entities = _entities(titles)
    by_name = {normalize(e["title"]): e["id"] for e in entities}
    owners: dict[str, list[Owns]] = defaultdict(list)
    owns: list[Owns] = []
    for doc in docs:
        eid = by_name[normalize(doc.title)]
        owner: Owns = {"id": "o:" + digest([doc.id, eid]), "paragraph_id": doc.id, "entity_id": eid}
        owns.append(owner)
        owners[eid].append(owner)

    origins: dict[tuple[str, str], set[str]] = defaultdict(set)
    for rows in owners.values():
        for left, right in combinations(rows, 2):
            origins[left["paragraph_id"], right["paragraph_id"]].update([left["id"], right["id"]])

    patterns = [
        (e["id"], alias, re.compile(r"(?<!\w)(?=(" + re.escape(alias) + r")(?!\w))"))
        for e in entities
        for alias in e["aliases"]
    ]
    mentions: list[Mention] = []
    for doc in docs:
        searchable, spans = _search_text(doc.text)
        for eid, alias, pattern in patterns:
            for match in pattern.finditer(searchable):
                intervals = spans[match.start(1) : match.end(1)]
                start = min(i for i, _ in intervals)
                end = max(j for _, j in intervals)
                if (start and WORD_CHARACTER.match(doc.text, start - 1)) or WORD_CHARACTER.match(
                    doc.text, end
                ):
                    continue
                exact = doc.text[start:end]
                if normalize(exact) != alias:
                    continue  # A compatible character may expand across a word boundary.
                edge_id = "m:" + digest([doc.id, eid, start, end, alias])
                mentions.append(
                    {
                        "id": edge_id,
                        "paragraph_id": doc.id,
                        "entity_id": eid,
                        "start": start,
                        "end": end,
                        "text": exact,
                        "alias": alias,
                    }
                )
                for owner in owners[eid]:
                    other = owner["paragraph_id"]
                    if doc.id != other:
                        pair = min(doc.id, other), max(doc.id, other)
                        origins[pair].update([edge_id, owner["id"]])
    projection: list[Link] = [
        {"source": a, "target": b, "evidence_edge_ids": sorted(edge_ids)}
        for (a, b), edge_ids in sorted(origins.items())
    ]
    mentions.sort(key=lambda m: (m["paragraph_id"], m["start"], m["end"], m["entity_id"]))
    return {
        "paragraphs": [Paragraph(id=d.id, title=d.title, text=d.text) for d in docs],
        "entities": entities,
        "owns": owns,
        "mentions": mentions,
        "projection": projection,
    }
