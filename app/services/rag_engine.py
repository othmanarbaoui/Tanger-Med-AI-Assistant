"""
Moteur RAG principal.
Encapsule : retriever MMR + reranker cross-encoder + pipeline LangChain.
"""
import logging
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from pydantic import Field
from sentence_transformers import CrossEncoder

from app.config import cfg
from app.utils.helpers import format_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
Tu es un assistant expert en analyse documentaire.

Ta mission est de répondre aux questions en te basant EXCLUSIVEMENT
sur les extraits de documents fournis ci-dessous.

Règles absolues :
1. N'utilise JAMAIS de connaissance externe au contexte fourni.
2. Ne mentionne jamais le contexte, les documents ou les sources dans ta réponse.
3. Réponds de manière directe et naturelle, comme si tu connaissais déjà l'information.
4. Structure ta réponse clairement : commence par le point principal, puis les détails.
5. Si plusieurs informations pertinentes existent, synthétise-les en une réponse cohérente.
6. Si l'information est absente du contexte, réponds EXACTEMENT :
   "Je ne trouve pas cette information dans ma base de connaissances."
7. Ne commence JAMAIS par : "Selon le contexte", "D'après les documents",
   "Le contexte indique", "Les extraits mentionnent".

Format de réponse :
- Réponse concise et structurée
- Utilise des listes à puces pour les énumérations (max 5 points)
- Longueur adaptée à la complexité de la question
"""

HUMAN_PROMPT = """\
Contexte documentaire :
──────────────────────
{context}
──────────────────────

Question : {question}
"""

# ---------------------------------------------------------------------------
# Retriever MMR + Reranker
# ---------------------------------------------------------------------------
class RerankerRetriever(BaseRetriever):
    """
    Retriever deux étapes :
    1. MMR FAISS  → top-k candidats diversifiés
    2. Cross-encoder → reranking précis → top-n final
    """
    vectorstore: Any
    cross_encoder: Any = Field(default=None)
    retrieval_k: int = cfg.retrieval_k
    rerank_top_n: int = cfg.rerank_top_n
    mmr_lambda: float = cfg.mmr_lambda

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        candidates = self.vectorstore.max_marginal_relevance_search(
            query, k=self.retrieval_k, lambda_mult=self.mmr_lambda
        )
        if not candidates:
            return []
        if self.cross_encoder is None:
            return candidates[: self.rerank_top_n]

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: self.rerank_top_n]]


# ---------------------------------------------------------------------------
# RAGEngine — façade principale
# ---------------------------------------------------------------------------
class RAGEngine:
    """
    Façade unifiée : initialise et expose ask() + get_sources().
    Utilise @st.cache_resource pour ne pas se réinstancier à chaque requête.
    """

    def __init__(self, vectorstore: FAISS, api_key: str):
        cross_encoder = CrossEncoder(cfg.reranker_model)

        self.retriever = RerankerRetriever(
            vectorstore=vectorstore,
            cross_encoder=cross_encoder,
        )

        llm = ChatGoogleGenerativeAI(
            model=cfg.llm_model,
            temperature=cfg.llm_temperature,
            max_output_tokens=cfg.llm_max_tokens,
            google_api_key=api_key,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])

        self._chain = (
            {
                "context": self.retriever | RunnableLambda(format_context),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        """Retourne la réponse du LLM."""
        return self._chain.invoke(question)

    def get_sources(self, question: str) -> list[dict]:
        """Retourne les sources sans doublons (fichier + page)."""
        docs = self.retriever.invoke(question)
        seen, sources = set(), []
        for doc in docs:
            key = (
                doc.metadata.get("source_file", "?"),
                doc.metadata.get("page", "?"),
            )
            if key not in seen:
                seen.add(key)
                sources.append({"file": key[0], "page": key[1]})
        return sources
