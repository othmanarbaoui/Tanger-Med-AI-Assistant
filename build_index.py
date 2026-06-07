"""
Script offline : construire le vectorstore FAISS depuis vos PDF.

Usage :
    python build_index.py --data Data/ --out app/artifacts/

À exécuter UNE SEULE FOIS en local, puis commiter app/artifacts/ sur GitHub.
"""
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from app.services.vectorstore import build_vectorstore


def main():
    parser = argparse.ArgumentParser(description="Construire l'index FAISS")
    parser.add_argument("--data", default="Data", help="Dossier contenant les PDF")
    parser.add_argument("--out", default="app/artifacts", help="Dossier de sortie du vectorstore")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)

    print(f"\n🔨 Construction de l'index FAISS")
    print(f"   Source  : {data_dir}")
    print(f"   Sortie  : {out_dir}\n")

    store = build_vectorstore(data_dir, out_dir)
    print(f"\n✅ Index construit — {store.index.ntotal} vecteurs")
    print(f"   Committez 'app/artifacts/' sur GitHub pour le déploiement.")


if __name__ == "__main__":
    main()
