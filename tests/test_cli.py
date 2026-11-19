import json
import os
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures/bridge.json"


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "graphmind.cli", *map(str, args)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_query_case_file_and_export(tmp_path):
    question = tmp_path / "q.txt"
    question.write_text("indigo")
    out = tmp_path / "result.json"
    result = cli("query", FIXTURE, "--question-file", question, "--output", out)
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["ranking"][0]["id"] == "a"
    assert json.loads(cli("query", FIXTURE).stdout)["graph"]["mentions"]
    assert json.loads(cli("export", FIXTURE).stdout)["statements"]


def test_cli_validation_and_atomic_non_clobber(tmp_path):
    out = tmp_path / "saved"
    out.write_text("original")
    for args in [("--mix", "nan"), (), ("--depth", "7")]:
        result = cli("query", FIXTURE, "--output", out, *args)
        assert result.returncode != 0 and "Traceback" not in result.stderr
        assert out.read_text() == "original"
    bad = tmp_path / "invalid.json"
    bad.write_text("[1]")
    new = tmp_path / "new"
    failed = cli("query", bad, "--question", "where", "--output", new)
    assert failed.returncode != 0 and not new.exists()


def test_simultaneous_writers_commit_one_complete_result(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    output = tmp_path / "race.json"
    with ThreadPoolExecutor(2) as pool:
        results = list(
            pool.map(
                lambda term: cli("query", FIXTURE, "-q", term, "-o", output), ["granite", "indigo"]
            )
        )
    assert sorted(r.returncode for r in results) == [0, 2]
    assert json.loads(output.read_text())["ranking"][0]["id"] in {"a", "d"}
    assert list(tmp_path.iterdir()) == [output]


def test_stdout_is_utf8_under_an_ascii_stream_encoding(tmp_path):
    corpus = tmp_path / "corpus.json"
    title = "Café"
    corpus.write_text(json.dumps([{"id": "p", "title": title, "text": ""}]), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "graphmind.cli", "export", str(corpus)],
        env={**os.environ, "PYTHONIOENCODING": "ascii"},
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert title.encode("utf-8") in result.stdout
    statements = json.loads(result.stdout.decode("utf-8"))["statements"]
    assert statements[0]["parameters"]["rows"][0]["title"] == title


def test_question_file_limit_counts_original_crlf_codepoints(tmp_path):
    question = tmp_path / "q.txt"
    question.write_bytes(b"qq" + b"\r\n" * 999)
    valid = cli("query", FIXTURE, "--question-file", question)
    assert valid.returncode == 0, valid.stderr
    question.write_bytes(b"q" + b"\r\n" * 1000)
    invalid = cli("query", FIXTURE, "--question-file", question)
    assert invalid.returncode == 2
    assert json.loads(invalid.stderr)["error"]["field"] == "question"
