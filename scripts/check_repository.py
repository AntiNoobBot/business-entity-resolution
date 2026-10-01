"""Read-only public-archive integrity and publication checks, using stdlib only."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    problems = []
    checked = 0
    excluded = {".git", "__pycache__", ".venv", "demo_output", "build", "dist"}
    forbidden = {".parquet", ".safetensors", ".npy", ".npz", ".bin", ".pt", ".pth", ".zip"}
    sensitive = [
        re.compile(r"/" + r"Users/[^/\s]+/"),
        re.compile(r"(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{20,}"),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"S[123]-[0-9]{5,}"),
        re.compile(r"(?m)^\s*S[123]\s+(?!entities in GT:|records matched to some).+\s\|"),
    ]
    for path in sorted(ROOT.rglob("*")):
        if set(path.relative_to(ROOT).parts) & excluded or not path.is_file():
            continue
        name = str(path.relative_to(ROOT))
        checked += 1
        if path.is_symlink() or path.suffix in forbidden or path.stat().st_size > 1024 * 1024:
            problems.append(f"Unexpected binary, large file, or symlink: {name}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"Non-text file: {name}")
            continue
        if any(pattern.search(text) for pattern in sensitive):
            problems.append(f"Review possible local path, credential, or real record identifier: {name}")
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=name)
            except SyntaxError as exc:
                problems.append(f"Syntax error: {name}:{exc.lineno}")
        if path.suffix == ".json":
            try:
                json.loads(text)
            except ValueError:
                problems.append(f"Invalid JSON: {name}")
        if path.suffix == ".md":
            for link in re.findall(r"\]\(([^\s)]+)\)", text):
                if "://" in link or link.startswith("#"):
                    continue
                target = link.split("#", 1)[0]
                if target and not (path.parent / target).exists():
                    problems.append(f"Broken relative link in {name}: {target}")
    archive = json.loads((ROOT / "reports/manifests/source_archive.json").read_text())
    for item in archive:
        path = ROOT / item["destination"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            problems.append(f"Archived source/evidence checksum mismatch: {item['destination']}")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"PASS: {checked} public files; {len(archive)} archived checksums; Python syntax, JSON, links, and artifact scan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
