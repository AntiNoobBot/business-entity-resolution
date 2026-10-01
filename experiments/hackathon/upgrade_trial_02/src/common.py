"""Integrity and atomic writes for the dual-machine retrieval handoff."""
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_relative(root, name):
    root = Path(root).resolve()
    result = (root / name).resolve()
    if result == root or root not in result.parents:
        raise ValueError(f"Unsafe relative path: {name}")
    return result


def verify_inputs(root):
    root = Path(root)
    manifest = read_json(root / "handoff_manifest.json")
    for name, expected in manifest["files"].items():
        path = safe_relative(root, name)
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Missing or changed transfer file: {name}")
    return manifest
