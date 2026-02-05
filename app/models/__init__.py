"""
Phase 4: Semantic Similarity Models
ML-based similarity computation between resume and job descriptions
"""

from .tfidf_model import TfidfSimilarityModel
from .embedding_model import EmbeddingSimilarityModel
from .similarity_engine import SimilarityEngine

__all__ = [
    "TfidfSimilarityModel",
    "EmbeddingSimilarityModel",
    "SimilarityEngine",
]
