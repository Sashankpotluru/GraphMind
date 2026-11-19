"""Parameterized Neo4j export for a single graph; execution belongs to the caller."""

from typing import TypedDict

from .graph import Graph


class Statement(TypedDict):
    query: str
    parameters: dict[str, object]


class Export(TypedDict):
    statements: list[Statement]


def neo4j_export(graph: Graph) -> Export:
    statements: list[Statement] = []

    def add(query: str, rows: object) -> None:
        statements.append({"query": "UNWIND $rows AS r " + query, "parameters": {"rows": rows}})

    add(
        "MERGE (p:Paragraph {id: r.id}) SET p.title = r.title, p.text = r.text", graph["paragraphs"]
    )
    add(
        "MERGE (e:Entity {id: r.id}) SET e.title = r.title, e.aliases = r.aliases",
        graph["entities"],
    )
    endpoints = "MATCH (p:Paragraph {id: r.paragraph_id}), (e:Entity {id: r.entity_id}) "
    add(endpoints + "MERGE (p)-[:OWNS {id: r.id}]->(e)", graph["owns"])
    add(
        endpoints + "MERGE (p)-[m:MENTIONS {id: r.id}]->(e) "
        "SET m.start = r.start, m.end = r.end, m.text = r.text, m.alias = r.alias",
        graph["mentions"],
    )
    add(
        "MATCH (a:Paragraph {id: r.source}), (b:Paragraph {id: r.target}) "
        "MERGE (a)-[p:PROJECTED]->(b) SET p.evidence_edge_ids = r.evidence_edge_ids",
        graph["projection"],
    )
    return {"statements": statements}
