import json
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
