# GraphMind engine contract

GraphMind ranks a supplied set of candidate paragraphs. A link records a lexical title mention or shared ownership of a title. It can connect relevant evidence, or connect paragraphs that happen to use the same words. It does not establish a semantic relation or verify an answer.

## Input

Import `Document`, `document_id`, `build_graph`, `bm25`, `expand`, `retrieve` and `neo4j_export` from `graphmind`. The runtime uses Python 3.12+ and the standard library.

`Document(id, title, text)` is a frozen dataclass. A corpus is a sequence of 1–100 such objects or mappings with exactly those three string fields. IDs must be unique within a corpus. IDs and titles must contain a non-whitespace character. Empty bodies are valid. Each body may contain up to 50,000 Unicode code points, with at most 1,000,000 across bodies. Strings must encode as UTF-8. The original strings are preserved. The question must contain at least one non-whitespace character and at most 2,000 code points including surrounding whitespace.

Call `document_id(title, text)` to generate the stable paragraph ID used by data adapters. It returns the full lowercase SHA-256 hex digest of UTF-8 compact JSON `[title,text]`, with non-ASCII characters emitted directly and no normalization. The core also accepts caller-assigned IDs. It rejects duplicates. Dataset deduplication, source-index mappings and changes to support counts belong to the adapter.

## Graph

`build_graph(documents)` takes only the corpus. Each array below is present even when empty. In this and subsequent schemas, `str` is a JSON string and `float` a finite JSON number.

| Key | Array element |
| --- | --- |
| `paragraphs` | `{id: str, title: str, text: str}` |
| `entities` | `{id: str, title: str, aliases: [str]}` |
| `owns` | `{id: str, paragraph_id: str, entity_id: str}` |
| `mentions` | `{id: str, paragraph_id: str, entity_id: str, text: str, start: int, end: int, alias: str}` |
| `projection` | `{source: str, target: str, evidence_edge_ids: [str]}` |

Each paragraph owns one canonical entity. Title identity applies NFKC, casefolding and whitespace collapse, then removes outer whitespace. Equivalent titles share an entity whose display title is the lexicographically smallest original spelling. The complete normalized title always serves as an alias. A balanced trailing parenthetical qualifier yields a shorter alias only if it contains three or more letters, is absent from the committed English stopword set, and identifies exactly one entity including full-title owners. For example, two qualified Mercury titles prevent a bare Mercury link to either. A separately owned full title Mercury keeps its own exact matches.

Mention extraction searches bodies, including a paragraph's mentions of its own entity. In both the normalized text and original source a match starts after a non-word character or at the beginning, and ends before a non-word character or at the end. Word characters follow Python Unicode regular expression `\w`. Each eligible full alias is matched independently, including overlaps. Every match maps back through Unicode normalization, casefold expansion and whitespace collapse to its exact source slice. A partial compatibility expansion that cannot be reconstructed from its source slice is discarded.

For each mention, `paragraph.text[start:end] == mention.text`, where `end` is exclusive. Offsets count original Unicode code points. JavaScript can take the same slice using `Array.from(text).slice(start, end).join("")`. `alias` is the normalized string matched. It may differ from the source spelling and length.

Entity IDs use the prefix `e:` plus the digest of `[normalized_title]`. Ownership and mention IDs use `o:` plus the digest of `[paragraph_id,entity_id]` and `m:` plus the digest of `[paragraph_id,entity_id,start,end,alias]`, respectively. Each digest uses the compact JSON convention above. Node type distinguishes paragraph IDs from entity IDs.

Projection connects a mentioning paragraph to each different paragraph that owns the matched entity. It also connects distinct paragraphs that own the same entity. Two paragraphs that merely mention the same entity do not gain a direct edge for that reason. For each unordered pair there is one edge, oriented in storage with `source < target`. Its sorted unique evidence list contains the mention and ownership IDs for each mention–owner connection and both ownership IDs for a shared-title connection. Repeated mentions add provenance and leave the neighbor transition weight unchanged. Reordering candidate input leaves all graph arrays and score outputs stable.

## Scoring

`bm25(documents, question)` returns `{paragraph_id: raw_score}` for every paragraph. Each document contributes its title, a newline and its body. Tokenization extracts Unicode word runs, casefolds each token and removes terms in `graphmind.documents.STOPWORDS`. There is no stemming. Length is the number of retained tokens. With `k1 = 1.2` and `b = 0.75`, each distinct retained query term contributes once:

```text
idf = log(1 + (N - df + 0.5) / (df + 0.5))
contribution = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / mean_length))
```

`expand(graph, base_scores, depth=2, mix=0.35, seeds=2)` accepts a graph produced by `build_graph` and a mapping from known paragraph IDs to finite, nonnegative scores. Omitted IDs have score zero. Booleans and numeric strings are invalid. Depth is an integer from 0 through 3, mix ranges from 0 through 1, and seeds is an integer from 1 through 3. Public input validation raises `GraphMindError` (a `ValueError`) with `field` and `detail` attributes. A caller supplying graph data should use the `build_graph` schema and invariants.

The base distribution normalizes all raw scores by their sum. The top positive seeds, ordered by descending raw score then ascending ID, are separately normalized to sum to one. At each hop, mass at a paragraph splits equally among unique neighbors. A dangling node retains its mass. For hop t through the requested depth h, the walk weights are `0.5**t / sum(0.5**i for i in range(h+1))`. Both the graph and every hop use all paragraph IDs.

| Expansion key | Value and meaning |
| --- | --- |
| `ranking` | All `{id: str, score: float, base_score: float, graph_score: float}` rows sorted by descending final score then ascending ID. Base and graph scores are pre-mix probabilities. |
| `hops` | Array of `{paragraph_id: float}` maps, ordered from p0 through ph, including the initial seed distribution. |
| `per_seed` | `{seed_id: {paragraph_id: float}}`. Each map already includes that seed's initial mass and the normalized hop weights. Sum across seeds to obtain each pre-mix graph score. |
| `seed_weights` | `{seed_id: float}` with the selected seeds' initial masses. |
| `depth`, `mix`, `seeds` | The requested integer depth, numeric mixing coefficient and integer seed count. |
| `effective_mix` | The applied coefficient. Zero if depth is zero, mix is zero, there is no projection edge or all base scores are zero. Otherwise it equals `mix`. |

The final score is `(1 - effective_mix) * base_score + effective_mix * graph_score`. When expansion is bypassed, the normalized base is returned exactly. Hops and per-seed contributions still describe the requested walk before mixing. With no positive scores all distributions and ranking scores are zero, and both seed maps are empty. Disconnected components receive walk mass only when they contain a seed. The function aggregates bounded distributions without enumerating paths.

`retrieve(documents, question, depth=2, mix=0.35, seeds=2)` returns all expansion keys plus `graph`. It builds the graph from corpus fields alone, scores the question with BM25, then expands those scores.

## CLI and authored example

```sh
graphmind query tests/fixtures/bridge.json --depth 2 --mix 1 --seeds 1
graphmind query tests/fixtures/bridge.json --question "Who hired weavers?"
graphmind query tests/fixtures/bridge.json --question-file question.txt --output result.json
graphmind export tests/fixtures/bridge.json --output neo4j.json
```

Input is a JSON array of documents or a case object with `documents` and an optional `question`. Additional case-level metadata is ignored by the graph builder. A query argument or UTF-8 question file overrides the embedded question. Successful commands emit UTF-8 JSON to stdout or create a new output file. Atomic file publication uses a sibling temporary file and a hard link, so the target filesystem must support hard links. An existing target is preserved. Data and I/O failures return exit code 2 and a JSON stderr object `{"error":{"field":str,"detail":str}}`. Argument syntax errors use argparse diagnostics.

The authored fixture's seed paragraph mentions Bell Harbor, which mentions Rowan Vale. After two hops, half the seed mass has returned to the starting paragraph and half reaches Rowan Vale. The disconnected quarry receives zero. At depth two with mix one, the combined hop score for Rowan Vale is 1/14. This path illustrates connectivity. The score includes every eligible walk, and a connected paragraph may still fail to answer the question.

## Neo4j boundary

`neo4j_export(graph)` returns `{statements: [{query: str, parameters: {rows: [...]}}]}` with five ordered batches for paragraphs, entities, OWNS, MENTIONS and PROJECTED links. Each rows array has exactly the corresponding graph schema above. All user values travel through the `$rows` parameter. Run the statements in order against a dedicated graph if importing them. The stored direction of PROJECTED follows canonical ID order. Use it as an undirected relationship for traversal. The exporter generates statements for a graph snapshot. Deleting stale data and orchestrating database transactions belong to the importer. Runtime retrieval uses the in-process graph. The export tests inspect the statement/parameter boundary without executing a Neo4j server.
