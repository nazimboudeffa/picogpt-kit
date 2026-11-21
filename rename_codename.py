"""Change le codename (nom de fleur) d'une release PicoGPT.

La source de vérité est ``project.json``. Le script :
1. met à jour ``project.json`` ;
2. remplace l'ancien nom dans les fichiers texte du dépôt (README, doc, fiche modèle) ;
3. ajoute une entrée au ``CHANGELOG.md`` ;
4. renomme le dossier de poids et le GGUF s'ils existent (sauf avec ``--no-data``).

Les scripts Python lisent ``config.py`` : ils n'ont plus de nom codé en dur.

Usage :
    python rename_codename.py --codename Iris
    python rename_codename.py --codename Iris --version 1.1.0 --dry-run
    python rename_codename.py --codename Iris --version 1.1.0 --no-data
"""

import argparse
import json
import re
from datetime import date
from pathlib import Path

from config import (
    CODENAME,
    FAMILY,
    HF_OWNER,
    MODEL_DIR,
    MODEL_NAME,
    PROJECT,
    PROJECT_FILE,
    RELEASE_TAG,
    ROOT,
    VERSION,
)

TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jinja", ".yml", ".yaml"}
EXCLUDED_FILES = {PROJECT_FILE.name, "vocab.json", "tokenizer.json"}
EXCLUDED_DIRS = {".git", ".venv", "__pycache__", MODEL_DIR}
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[\w.]+)?$")
CODENAME_RE = re.compile(r"^[A-Za-z][A-Za-z-]*$")


def text_files():
    """Parcourt les fichiers texte du dépôt, hors données du modèle et métadonnées."""
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS for part in relative.parts[:-1]):
            continue
        if path.name in EXCLUDED_FILES:
            continue
        yield path


def replace_in_texts(replacements, dry_run):
    """Remplace les anciens identifiants dans les fichiers texte et renvoie les comptes.

    Toutes les lectures sont faites avant les écritures : une erreur de lecture
    ne laisse pas le dépôt à moitié renommé.
    """
    updates = []
    for path in text_files():
        updated = path.read_text(encoding="utf-8")
        count = 0
        for old, new in replacements:
            count += updated.count(old)
            updated = updated.replace(old, new)
        if count:
            updates.append((path, updated, count))

    if not dry_run:
        for path, updated, _ in updates:
            path.write_text(updated, encoding="utf-8")

    return [(path.relative_to(ROOT), count) for path, _, count in updates]


def add_changelog_entry(version, codename, old_name, new_name, dry_run):
    """Insère une entrée de release avant la première entrée existante."""
    path = ROOT / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else "# Changelog\n\n"
    entry = (
        f"## [{version}] — {codename} — {date.today().isoformat()}\n\n"
        f"Changement de codename : `{old_name}` → `{new_name}`.\n\n"
    )
    index = text.find("## [")
    if index == -1:
        text = text.rstrip() + "\n\n" + entry
    else:
        text = text[:index] + entry + text[index:]
    if not dry_run:
        path.write_text(text, encoding="utf-8")


def rename_data(old_name, new_name, dry_run):
    """Renomme le dossier de poids et le fichier GGUF s'ils sont présents."""
    pairs = [
        (ROOT / old_name, ROOT / new_name),
        (ROOT / f"{old_name}.gguf", ROOT / f"{new_name}.gguf"),
    ]
    for source, target in pairs:
        if not source.exists():
            continue
        if target.exists():
            print(f"! {target.name} existe déjà : déplacement ignoré.")
            continue
        prefix = "DRY-RUN — " if dry_run else ""
        print(f"{prefix}Renommage : {source.name} -> {target.name}")
        if not dry_run:
            source.rename(target)


def main():
    parser = argparse.ArgumentParser(description="Change le codename d'une release PicoGPT")
    parser.add_argument("--codename", required=True, help="nouveau nom de fleur (ex. Iris)")
    parser.add_argument("--version", default=VERSION, help=f"nouvelle version (défaut : {VERSION})")
    parser.add_argument("--family", default=FAMILY, help=f"famille de modèles (défaut : {FAMILY})")
    parser.add_argument("--hf-owner", default=HF_OWNER, help=f"propriétaire HF (défaut : {HF_OWNER})")
    parser.add_argument("--no-data", action="store_true",
                        help="ne pas renommer le dossier de poids ni le GGUF")
    parser.add_argument("--dry-run", action="store_true",
                        help="affiche les changements sans rien écrire")
    args = parser.parse_args()

    codename = args.codename.strip().capitalize()
    if not CODENAME_RE.fullmatch(codename):
        raise SystemExit("Codename invalide : lettres et tirets uniquement (ex. Iris, Bell-flower).")
    if not VERSION_RE.fullmatch(args.version):
        raise SystemExit("Version invalide : attendu x.y.z (ex. 1.1.0).")

    old_name = f"{FAMILY}-{CODENAME}"
    new_name = f"{args.family}-{codename}"
    new_tag = f"v{args.version}-{codename.lower()}"

    if (old_name == new_name and args.version == VERSION
            and args.hf_owner == HF_OWNER):
        raise SystemExit("Aucun changement : le codename, la version et le propriétaire sont identiques.")

    print(f"Release actuelle : {old_name} — v{VERSION} ({HF_OWNER})")
    print(f"Nouvelle release : {new_name} — v{args.version} ({args.hf_owner})")
    print(f"Tag git suggéré  : {new_tag}")

    replacements = [
        (old_name, new_name),
        (RELEASE_TAG, new_tag),
        (f'"codename": "{CODENAME}"', f'"codename": "{codename}"'),
        (f'"version": "{VERSION}"', f'"version": "{args.version}"'),
        (f"codename: {CODENAME.lower()}", f"codename: {codename.lower()}"),
        (f"version: {VERSION}", f"version: {args.version}"),
        (f"codename {CODENAME}", f"codename {codename}"),
    ]
    if args.hf_owner != HF_OWNER:
        replacements.append((f"{HF_OWNER}/", f"{args.hf_owner}/"))

    if args.dry_run:
        print("\nDRY-RUN : aucun fichier ne sera modifié.\n")

    updated = dict(PROJECT)
    updated.update(
        family=args.family,
        version=args.version,
        codename=codename,
        hf_owner=args.hf_owner,
    )
    if not args.dry_run:
        PROJECT_FILE.write_text(
            json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"{'DRY-RUN — ' if args.dry_run else ''}Mise à jour de project.json")

    changed = replace_in_texts(replacements, args.dry_run)
    print(f"Fichiers texte mis à jour : {len(changed)}")
    for path, count in changed:
        print(f"  {path} ({count} remplacement{'s' if count > 1 else ''})")

    add_changelog_entry(args.version, codename, old_name, new_name, args.dry_run)
    print(f"{'DRY-RUN — ' if args.dry_run else ''}Entrée ajoutée au CHANGELOG.md")

    if not args.no_data:
        rename_data(old_name, new_name, args.dry_run)
    else:
        print("--no-data : poids et GGUF conservés sous leur ancien nom.")

    if not args.dry_run:
        print("\nVérifications et publication :")
        print("  python config.py")
        print("  python -m py_compile config.py rename_codename.py train.py test.py serve.py evaluate.py export_gguf.py publish_hf.py")
        print("  python export_gguf.py && python publish_hf.py")
        print(f"  git tag -a {new_tag} -m \"Release {new_name}\" && git push origin {new_tag}")
        print("\nPensez à mettre à jour le tableau comparatif du MODEL_CARD.md si la fleur précédente change.")


if __name__ == "__main__":
    main()
