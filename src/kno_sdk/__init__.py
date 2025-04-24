"""KNO-SDK — clone, index and search GitHub repos via Chroma embeddings."""

from .agent import search, RepoIndex, EmbeddingMethod, agent_query, LLMProvider
from .embedding import clone_and_index
__all__ = [
    "clone_and_index",
    "search",
    "RepoIndex",
    "EmbeddingMethod",
    "agent_query",
    "LLMProvider"
]