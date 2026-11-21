"""Configuration partagée du projet.

La source de vérité est ``project.json`` : les scripts importent les
constantes définies ici au lieu de coder le nom de la fleur en dur.
Pour changer de codename, utilisez ``python rename_codename.py``.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_FILE = ROOT / "project.json"

with PROJECT_FILE.open(encoding="utf-8") as _file:
    PROJECT = json.load(_file)

FAMILY = PROJECT["family"]
VERSION = PROJECT["version"]
CODENAME = PROJECT["codename"]
HF_OWNER = PROJECT["hf_owner"]

MODEL_NAME = f"{FAMILY}-{CODENAME}"
MODEL_DIR = MODEL_NAME
GGUF_FILE = f"{MODEL_NAME}.gguf"
HF_REPO_ID = f"{HF_OWNER}/{MODEL_NAME}"
RELEASE_TAG = f"v{VERSION}-{CODENAME.lower()}"


def summary():
    """Renvoie une description courte de la release courante."""
    return f"{MODEL_NAME} — v{VERSION} ({RELEASE_TAG}) — {HF_REPO_ID}"


if __name__ == "__main__":
    print(summary())
