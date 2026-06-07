"""
Utilitaires de nettoyage de texte pour les PDF.
"""
import re
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Supprime les artefacts courants des PDF."""
    text = re.sub(r'\n{3,}', '\n\n', text)   # max 2 sauts de ligne consécutifs
    text = re.sub(r'[ \t]+', ' ', text)        # espaces/tabulations multiples
    text = re.sub(r' \n', '\n', text)          # espace superflu avant saut de ligne
    text = re.sub(r'\f', '\n', text)           # form feed → saut de ligne
    return text.strip()


def format_context(docs: list[Document]) -> str:
    """
    Formate les documents récupérés en bloc de contexte structuré
    pour le prompt LLM.
    """
    sections = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "source inconnue")
        page = doc.metadata.get("page", "?")
        sections.append(
            f"[Extrait {i} | {source} | page {page}]\n{doc.page_content}"
        )
    return "\n\n".join(sections)
