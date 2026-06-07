"""
Interface Streamlit du système RAG.
Déploiement : Streamlit Community Cloud (clé API via Secrets).
"""
import streamlit as st
from pathlib import Path

# ── Configuration de la page ──────────────────────────────────────────────
st.set_page_config(
    page_title="Système RAG",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Imports internes ──────────────────────────────────────────────────────
from app.config import cfg
from app.services.vectorstore import load_vectorstore
from app.services.rag_engine import RAGEngine


# ── Chargement avec cache (une seule fois par session serveur) ────────────
@st.cache_resource(show_spinner="⏳ Chargement du modèle et de l'index…")
def load_engine() -> RAGEngine:
    """
    Charge le vectorstore et instancie le moteur RAG.
    La clé API est lue depuis st.secrets (Streamlit Cloud)
    ou depuis les variables d'environnement (local).
    """
    # Priorité : Streamlit Secrets → variable d'environnement
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        import os
        api_key = os.getenv("GOOGLE_API_KEY", "")

    if not api_key:
        st.error(
            "🔑 Clé API Google manquante.\n\n"
            "En local : créez un fichier `.env` avec `GOOGLE_API_KEY=…`\n"
            "Sur Streamlit Cloud : ajoutez la clé dans **Settings → Secrets**."
        )
        st.stop()

    vectorstore = load_vectorstore(cfg.artifacts_dir)
    return RAGEngine(vectorstore=vectorstore, api_key=api_key)


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Système RAG")
    st.markdown(
        """
        Posez vos questions sur la base documentaire.

        **Modèle LLM** : Gemini 2.0 Flash  
        **Embedding** : multilingual-E5-large  
        **Retrieval** : MMR + Cross-Encoder Reranker  
        """
    )
    show_sources = st.toggle("Afficher les sources", value=True)
    st.divider()
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

# ── Titre principal ───────────────────────────────────────────────────────
st.title("💬 Assistant Documentaire")
st.caption("Interrogez vos documents en langage naturel")

# ── Initialisation de l'historique de conversation ───────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Affichage de l'historique ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s['file']}` — page {s['page']}")

# ── Champ de saisie ───────────────────────────────────────────────────────
if question := st.chat_input("Votre question…"):
    # Afficher la question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Charger le moteur (depuis le cache)
    engine = load_engine()

    # Générer la réponse
    with st.chat_message("assistant"):
        with st.spinner("Recherche et génération en cours…"):
            response = engine.ask(question)
            sources = engine.get_sources(question) if show_sources else []

        st.markdown(response)

        if show_sources and sources:
            with st.expander("📚 Sources"):
                for s in sources:
                    st.markdown(f"- `{s['file']}` — page {s['page']}")

    # Sauvegarder dans l'historique
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
    })
