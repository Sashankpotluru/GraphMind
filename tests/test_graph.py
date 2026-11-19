import json

from graphmind import Document, build_graph, neo4j_export


def test_aliases_and_original_offsets():
    source = "🚤 Großstraße; GROSSSTRASSE. XGrossstrasse StraßeS. ALPHA\t\nBETA. Cafe\u0301!"
    docs = [
        Document("a", "Journal", source),
        Document("b", "Großstraße", "x"),
        Document("c", "Alpha Beta (boat)", "x"),
        Document("d", "Café", "x"),
    ]
    graph = build_graph(docs)
    assert graph == build_graph(docs[::-1])
    mentions = [m for m in graph["mentions"] if m["paragraph_id"] == "a"]
    actual = {(m["text"], m["alias"]) for m in mentions}
    assert actual == {
        ("Großstraße", "grossstrasse"),
        ("GROSSSTRASSE", "grossstrasse"),
        ("ALPHA\t\nBETA", "alpha beta"),
        ("Cafe\u0301", "café"),
    }
    assert all(source[m["start"] : m["end"]] == m["text"] for m in mentions)
    assert next(m["start"] for m in mentions if m["text"] == "Großstraße") == 2


def test_ambiguous_short_and_generic_stripped_aliases():
    titles = ["Mercury (planet)", "Mercury (element)", "Al (name)", "The (film)"]
    body = "Mercury, Al, the. Mercury (planet) and MERCURY (element)!"
    graph = build_graph(
        [Document("q", "Source", body)] + [Document(str(i), t, "x") for i, t in enumerate(titles)]
    )
    assert {m["alias"] for m in graph["mentions"]} == {"mercury (planet)", "mercury (element)"}
    assert len(graph["entities"]) == 5


def test_projection_unifies_repeated_mentions_and_shared_owners():
    docs = [
        Document("a", "Journal", "Bell Harbor visits Bell Harbor."),
        Document("b", "Bell Harbor", "Bell Harbor."),
        Document("c", "Ｂｅｌｌ  Harbor", "x"),
        Document("d", "Éric's log", "x"),
    ]
    graph = build_graph(docs)
    entities = {e["id"]: e for e in graph["entities"]}
    own = {o["paragraph_id"]: o for o in graph["owns"]}
    assert own["b"]["entity_id"] == own["c"]["entity_id"] in entities
    edges = {(e["source"], e["target"]): set(e["evidence_edge_ids"]) for e in graph["projection"]}
    assert set(edges) == {("a", "b"), ("a", "c"), ("b", "c")}
    origins = {m["id"] for m in graph["mentions"] if m["paragraph_id"] == "a"}
    assert len(origins) == 2
    assert edges["a", "c"] == origins | {own["c"]["id"]}
    assert {own["b"]["id"], own["c"]["id"]} <= edges["b", "c"]


def test_export_uses_parameters_for_untrusted_text():
    payload = "Éric's `notes` \\ 🛶"
    graph = build_graph([Document("x'", payload, payload)])
    statements = neo4j_export(graph)["statements"]
    assert all("$rows" in s["query"] and payload not in s["query"] for s in statements)
    assert payload in json.dumps([s["parameters"] for s in statements], ensure_ascii=False).replace(
        "\\\\", "\\"
    )
    assert {m["id"] for m in graph["mentions"]} <= {
        row["id"] for s in statements for row in s["parameters"]["rows"] if "id" in row
    }


def test_nested_disambiguator_and_partial_unicode_expansion():
    docs = [
        Document("s", "Log", "Cirrus landed. A fraction ½ is here."),
        Document("a", "Cirrus (aircraft (prototype))", "x"),
        Document("b", "1", "x"),
        Document("c", "1⁄2", "x"),
    ]
    mentions = build_graph(docs)["mentions"]
    assert {(m["text"], m["alias"]) for m in mentions} == {("Cirrus", "cirrus"), ("½", "1⁄2")}


def test_nfkc_composition_and_fold_have_independent_source_spans():
    docs = [
        Document("s", "Log", "📚 가 ＡＢＣ A\u0327\u0301."),
        Document("h", "가", ""),
        Document("w", "abc", ""),
        Document("m", "Á\u0327", ""),
    ]
    mentions = build_graph(docs)["mentions"]
    assert {(m["start"], m["end"], m["text"]) for m in mentions} == {
        (2, 4, "가"),
        (5, 8, "ＡＢＣ"),
        (9, 12, "A\u0327\u0301"),
    }


def test_casefold_cannot_create_a_new_boundary_inside_source_word():
    docs = [Document("s", "Log", "İBar sails. Bar returns."), Document("b", "Bar", "")]
    mentions = build_graph(docs)["mentions"]
    assert [(m["start"], m["end"]) for m in mentions] == [(12, 15)]
