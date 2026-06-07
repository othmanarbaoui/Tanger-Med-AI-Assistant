"""
Construction et chargement du VectorStore FAISS.
Le vectorstore est pré-construit (offline) et versionné dans app/artifacts/.
"""
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from app.config import cfg
from app.utils.helpers import clean_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embeddings wrapper E5 (préfixes obligatoires pour ce modèle)
# ---------------------------------------------------------------------------
class E5Embeddings(HuggingFaceEmbeddings):
    """Ajoute automatiquement les préfixes query:/passage: requis par E5."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")


def get_embeddings() -> E5Embeddings:
    return E5Embeddings(
        model_name=cfg.embed_model,
        encode_kwargs={"normalize_embeddings": True, "batch_size": cfg.embed_batch_size},
        model_kwargs={"device": "cpu"},
    )


# ---------------------------------------------------------------------------
# Chargement (production : artifacts pré-indexés)
# ---------------------------------------------------------------------------
def load_vectorstore(store_dir: Path | None = None) -> FAISS:
    """
    Charge le vectorstore FAISS depuis app/artifacts/.
    Lève une erreur claire si l'index est absent.
    """
    store_dir = store_dir or cfg.artifacts_dir
    index_file = store_dir / "index.faiss"

    if not index_file.exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable dans '{store_dir}'.\n"
            "Exécutez d'abord : python scripts/build_index.py"
        )

    logger.info(f"Chargement vectorstore depuis '{store_dir}'")
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(store_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ---------------------------------------------------------------------------
# Construction offline (script séparé, pas appelé par Streamlit)
# ---------------------------------------------------------------------------
def build_vectorstore(data_dir: Path, store_dir: Path) -> FAISS:
    """
    Charge les PDF, découpe, encode et sauvegarde le vectorstore.
    À exécuter une seule fois en local avant de pousser sur GitHub.
    """
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Aucun PDF dans '{data_dir}'")

    # Chargement
    all_docs: list[Document] = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.page_content = clean_text(page.page_content)
            page.metadata["source_file"] = pdf_path.name
        all_docs.extend(pages)
        logger.info(f"  ✓ {pdf_path.name} ({len(pages)} pages)")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", "،", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= cfg.min_chunk_length]
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    logger.info(f"  ✂️  {len(chunks)} chunks créés")

    # Indexation
    embeddings = get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    store_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(store_dir))
    logger.info(f"  💾 VectorStore sauvegardé dans '{store_dir}'")
    return store
