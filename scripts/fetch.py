#!/usr/bin/env python3
"""Fetch a pinned numerical track into the Hugging Face cache and verify its files."""
import argparse
import fnmatch
import hashlib
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

TRACKS = {
    "tokenizer": (
        "thinkingmachines/Inkling-2-Mini",
        "ed40e7da625f594fc9c4990f5f66be24873c22d7",
        ("tokenizer.json",),
    ),
    "native": (
        "thinkingmachines/Inkling-2-Mini",
        "ed40e7da625f594fc9c4990f5f66be24873c22d7",
        ("*.json", "*.safetensors", "*.jinja", "tiktoken/*", "README.md"),
    ),
    "gguf": (
        "ggml-org/Inkling-2-Mini-GGUF",
        "e8a2cecf0e78914f07cc27b9be711a9b2c7561d9",
        ("Inkling-2-Mini-IQ4_NL.gguf", "mmproj-Inkling-2-Mini-BF16.gguf", "README.md"),
    ),
    "dspark-draft": (
        "unsloth/Inkling-2-Mini-GGUF",
        "ae9327c5fbc7333116ba32eee78bb5d2bbefae4f",
        ("dspark-Inkling-2-Mini-Q8_0.gguf",),
    ),
    "basalt": (
        "jesdga/Inkling-2-Mini-Basalt",
        "426fea7456c37c102be8a30d16e4f6bf0dcd892d",
        ("Inkling-2-Mini-NVFP4-q8.basalt", "LICENSE", "README.md"),
    ),
}


def fetch(track):
    repo, revision, patterns = TRACKS[track]
    files = HfApi().model_info(repo, revision=revision, files_metadata=True).siblings
    snapshot = Path(snapshot_download(repo, revision=revision, allow_patterns=list(patterns), max_workers=4))
    for file in files:
        if not any(fnmatch.fnmatchcase(file.rfilename, pattern) for pattern in patterns):
            continue
        path = snapshot / file.rfilename
        if path.stat().st_size != file.size:
            raise ValueError(f"Size mismatch for {path}")
        if file.lfs:
            with path.open("rb") as reader:
                if hashlib.file_digest(reader, "sha256").hexdigest() != file.lfs.sha256:
                    raise ValueError(f"SHA-256 mismatch for {path}")
        elif file.blob_id:
            digest = hashlib.sha1(f"blob {file.size}\0".encode())
            digest.update(path.read_bytes())
            if digest.hexdigest() != file.blob_id:
                raise ValueError(f"Git blob mismatch for {path}")
        print(f"verified {file.rfilename} ({file.size} bytes)", file=sys.stderr)
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", choices=TRACKS)
    print(fetch(parser.parse_args().track))
