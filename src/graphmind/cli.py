"""Command-line JSON input and atomic, create-only output."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .documents import GraphMindError
from .export import neo4j_export
from .graph import build_graph
from .retrieval import retrieve


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graphmind")
    commands = parser.add_subparsers(dest="command", required=True)
    query = commands.add_parser("query", help="rank paragraphs for a question")
    export = commands.add_parser("export", help="emit parameterized Neo4j statements")
    for command in (query, export):
        command.add_argument("input", type=Path, help="JSON documents array or case object")
        command.add_argument("--output", "-o", type=Path, help="create a new output file")
    questions = query.add_mutually_exclusive_group()
    questions.add_argument("--question", "-q")
    questions.add_argument("--question-file", type=Path)
    query.add_argument("--depth", type=int, default=2)
    query.add_argument("--mix", type=float, default=0.35)
    query.add_argument("--seeds", type=int, default=2)
    return parser


def _write_new(path: Path, data: str) -> None:
    # Linking a closed sibling file publishes complete bytes only if the name is free.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data.encode("utf-8"))
            handle.close()
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        case = payload if isinstance(payload, dict) else {}
        documents = case.get("documents") if isinstance(payload, dict) else payload
        if not isinstance(documents, list):
            raise GraphMindError("documents", "expected an array of paragraphs")
        if args.command == "export":
            result: object = neo4j_export(build_graph(documents))
        else:
            question = args.question if args.question is not None else case.get("question")
            if args.question_file is not None:
                question = args.question_file.read_bytes().decode("utf-8")
            if not isinstance(question, str):
                raise GraphMindError("question", "expected a nonblank string")
            result = retrieve(documents, question, args.depth, args.mix, args.seeds)
        data = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        if args.output:
            _write_new(args.output, data)
        else:
            sys.stdout.buffer.write(data.encode("utf-8"))
    except (ValueError, OSError) as exc:
        field = exc.field if isinstance(exc, GraphMindError) else "io"
        detail = exc.detail if isinstance(exc, GraphMindError) else str(exc)
        print(json.dumps({"error": {"field": field, "detail": detail}}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
