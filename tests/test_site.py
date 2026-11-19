import hashlib
import json
from zipfile import ZipFile

import pytest

from scripts.build_site import CORE_FILES, ROOT, bundle_core, copy_asset, site_inventory


def test_core_archive_is_deterministic_and_reuses_actual_python_sources(tmp_path):
    target = tmp_path / "engine.zip"
    bundle_core(target)
    original = target.read_bytes()
    bundle_core(target)
    assert target.read_bytes() == original
    with ZipFile(target) as archive:
        assert sorted(archive.namelist()) == sorted(f"graphmind/{name}" for name in CORE_FILES)
        for name in CORE_FILES:
            assert archive.read(f"graphmind/{name}") == (ROOT / "src/graphmind" / name).read_bytes()


def test_site_inventory_never_accepts_symlinks_or_changed_artifacts(tmp_path):
    public, output = tmp_path / "public", tmp_path / "output"
    public.mkdir()
    output.mkdir()
    content = b"published"
    (public / "record").write_bytes(content)
    row = {"path": "record", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    copy_asset(public, output, row)
    site_inventory(output, {"record"})
    assert json.loads((output / "site-manifest.json").read_text())["files"] == [row]
    (public / "record").write_text("tampered!")
    with pytest.raises(ValueError):
        copy_asset(public, output, row)
    (public / "linked").symlink_to(public)
    for name in ["../public/record", "linked/record", "/etc/hosts"]:
        with pytest.raises(ValueError):
            copy_asset(public, output, {**row, "path": name})


def test_private_or_missing_file_prevents_public_manifest(tmp_path):
    (tmp_path / "public").write_text("ok")
    (tmp_path / "cache").write_text("private")
    with pytest.raises(ValueError, match="allowlist"):
        site_inventory(tmp_path, {"public"})
    assert not (tmp_path / "site-manifest.json").exists()
    (tmp_path / "cache").unlink()
    with pytest.raises(ValueError, match="allowlist"):
        site_inventory(tmp_path, {"public", "missing"})
