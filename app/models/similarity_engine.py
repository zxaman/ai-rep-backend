"""
Similarity Engine - ML Model Controller
Orchestrates TF-IDF and Embedding models for comprehensive similarity scoring

This is the ML brain that combines multiple similarity approaches.

Interview Talking Point:
"I built a similarity engine that combines lexical and semantic models,
providing both explainable keyword matching and deep semantic understanding."
"""

from typing import Dict, List, Optional
from .tfidf_model import TfidfSimilarityModel
from .embedding_model import EmbeddingSimilarityModel
import logging

logger = logging.getLogger(__name__)


class SimilarityEngine:
    """
    Orchestrate multiple similarity models and combine their scores.
    
    This engine:
    1. Runs TF-IDF similarity (lexical/keyword matching)
    2. Runs embedding similarity (semantic meaning)
    3. Combines scores with configurable weights
    4. Provides comprehensive similarity analysis
    
    Why combine both models?
    - TF-IDF: Fast, explainable, catches exact keyword matches
    - Embeddings: Captures semantic meaning, understands synonyms
    - Combined: Best of both worlds
    
    Default weights:
    - TF-IDF: 30% (keyword importance)
    - Embeddings: 70% (semantic importance)
    """
    
    def __init__(
        self,
        tfidf_weight: float = 0.3,
        embedding_weight: float = 0.7,
        tfidf_params: Optional[Dict] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize similarity engine with model configurations.
        
        Args:
            tfidf_weight: Weight for TF-IDF similarity (0-1)
            embedding_weight: Weight for embedding similarity (0-1)
            tfidf_params: Optional parameters for TfidfSimilarityModel
            embedding_model_name: Name of sentence transformer model
        """
        # Validate weights
        if not (0 <= tfidf_weight <= 1):
            raise ValueError("tfidf_weight must be between 0 and 1")
        if not (0 <= embedding_weight <= 1):
            raise ValueError("embedding_weight must be between 0 and 1")
        
        # Normalize weights to sum to 1
        total_weight = tfidf_weight + embedding_weight
        if total_weight == 0:
            raise ValueError("At least one weight must be > 0")
        
        self.tfidf_weight = tfidf_weight / total_weight
        self.embedding_weight = embedding_weight / total_weight
        
        # Initialize models
        self.tfidf_model = TfidfSimilarityModel(**(tfidf_params or {}))
        self.embedding_model = EmbeddingSimilarityModel(model_name=embedding_model_name)
        
        logger.info(
            f"SimilarityEngine initialized with weights: "
            f"TF-IDF={self.tfidf_weight:.2f}, Embedding={self.embedding_weight:.2f}"
        )
    
    def compute_similarity(
        self,
        resume_text: str,
        job_text: str,
        include_details: bool = True
    ) -> Dict[str, any]:
        """
        Compute comprehensive similarity between resume and job description.
        
        This is the main method that combines both models.
        
        Args:
            resume_text: Cleaned resume text (from Phase 2)
            job_text: Job description or role template text
            include_details: Whether to include detailed results from each model
            
        Returns:
            {
                "semantic_similarity_score": float (0-1),  # ⭐ Main output
                "tfidf_similarity": float,
                "embedding_similarity": float,
                "tfidf_weight": float,
                "embedding_weight": float,
                "tfidf_details": {...},     # If include_details=True
                "embedding_details": {...}   # If include_details=True
            }
        """
        # Input validation
        if not resume_text or not resume_text.strip():
            return self._empty_result("Empty resume text")
        if not job_text or not job_text.strip():
            return self._empty_result("Empty job description text")
        
        # Compute TF-IDF similarity
        logger.debug("Computing TF-IDF similarity...")
        tfidf_result = self.tfidf_model.compute_similarity(resume_text, job_text)
        tfidf_score = tfidf_result.get("tfidf_similarity", 0.0)
        
        # Compute embedding similarity
        logger.debug("Computing embedding similarity...")
        embedding_result = self.embedding_model.compute_similarity(resume_text, job_text)
        embedding_score = embedding_result.get("embedding_similarity", 0.0)
        
        # Combine scores with weights
        final_score = (
            self.tfidf_weight * tfidf_score +
            self.embedding_weight * embedding_score
        )
        
        # Build result
        result = {
            "semantic_similarity_score": float(final_score),
            "tfidf_similarity": float(tfidf_score),
            "embedding_similarity": float(embedding_score),
            "tfidf_weight": self.tfidf_weight,
            "embedding_weight": self.embedding_weight,
        }
        
        # Add detailed results if requested
        if include_details:
            result["tfidf_details"] = tfidf_result
            result["embedding_details"] = embedding_result
        
        # Add interpretation
        result["interpretation"] = self._interpret_score(final_score)
        
        logger.info(
            f"Similarity computed: TF-IDF={tfidf_score:.3f}, "
            f"Embedding={embedding_score:.3f}, Final={final_score:.3f}"
        )
        
        return result
    
    def compute_batch_similarity(
        self,
        resume_texts: List[str],
        job_text: str,
        include_details: bool = False
    ) -> List[Dict[str, any]]:
        """
        Compute similarity for multiple resumes against one job description.
        
        More efficient than calling compute_similarity multiple times.
        
        Args:
            resume_texts: List of resume texts
            job_text: Job description text
            include_details: Whether to include detailed results
            
        Returns:
            List of similarity results for each resume
        """
        if not resume_texts:
            return []
        
        # Compute TF-IDF for all resumes
        tfidf_results = [
            self.tfidf_model.compute_similarity(resume, job_text)
            for resume in resume_texts
        ]
        
        # Compute embeddings in batch (more efficient)
        embedding_results = self.embedding_model.compute_batch_similarity(
            resume_texts,
            job_text
        )
        
        # Combine results
        results = []
        for i, (tfidf_res, emb_res) in enumerate(zip(tfidf_results, embedding_results)):
            tfidf_score = tfidf_res.get("tfidf_similarity", 0.0)
            emb_score = emb_res.get("embedding_similarity", 0.0)
            
            final_score = (
                self.tfidf_weight * tfidf_score +
                self.embedding_weight * emb_score
            )
            
            result = {
                "resume_index": i,
                "semantic_similarity_score": float(final_score),
                "tfidf_similarity": float(tfidf_score),
                "embedding_similarity": float(emb_score)
            }
            
            if include_details:
                result["tfidf_details"] = tfidf_res
                result["embedding_details"] = emb_res
            
            results.append(result)
        
        return results
    
    def explain_similarity(
        self,
        resume_text: str,
        job_text: str
    ) -> str:
        """
        Generate comprehensive human-readable explanation of similarity.
        
        Args:
            resume_text: Resume text
            job_text: Job description text
            
        Returns:
            Multi-line string with detailed similarity breakdown
        """
        result = self.compute_similarity(resume_text, job_text, include_details=True)
        
        if "error" in result:
            return f"Similarity Analysis: Unable to compute ({result['error']})"
        
        final_score = result["semantic_similarity_score"]
        tfidf_score = result["tfidf_similarity"]
        embedding_score = result["embedding_similarity"]
        
        tfidf_details = result.get("tfidf_details", {})
        embedding_details = result.get("embedding_details", {})
        
        # Build explanation
        explanation = f"""
═══════════════════════════════════════════════════════
         SEMANTIC SIMILARITY ANALYSIS
═══════════════════════════════════════════════════════

🎯 OVERALL SIMILARITY: {final_score:.1%}
{result['interpretation']}

───────────────────────────────────────────────────────
📊 MODEL BREAKDOWN
───────────────────────────────────────────────────────

1️⃣ TF-IDF Similarity (Keyword Matching)
   Score: {tfidf_score:.1%} (weight: {self.tfidf_weight:.0%})
   Shared Terms: {tfidf_details.get('shared_terms_count', 0)} keywords
   Top Keywords: {', '.join(tfidf_details.get('shared_terms', [])[:5])}

2️⃣ Embedding Similarity (Semantic Meaning)
   Score: {embedding_score:.1%} (weight: {self.embedding_weight:.0%})
   Model: {embedding_details.get('model_name', 'N/A')}
   Dimension: {embedding_details.get('embedding_dimension', 0)}

───────────────────────────────────────────────────────
💡 WHAT THIS MEANS
───────────────────────────────────────────────────────

Combined Score Formula:
  {final_score:.3f} = ({self.tfidf_weight:.2f} × {tfidf_score:.3f}) + ({self.embedding_weight:.2f} × {embedding_score:.3f})

• TF-IDF captures exact keyword matches (e.g., "Python", "SQL")
• Embeddings capture semantic meaning (e.g., "ML" ≈ "Machine Learning")
• Higher score = better alignment between resume and job requirements

───────────────────────────────────────────────────────
""".strip()
        
        return explanation
    
    def _interpret_score(self, score: float) -> str:
        """
        Interpret similarity score with human-readable description.
        
        Args:
            score: Similarity score (0-1)
            
        Returns:
            String interpretation
        """
        if score >= 0.75:
            return "🟢 Excellent Match - Strong alignment with job requirements"
        elif score >= 0.6:
            return "🟡 Good Match - Solid alignment with most requirements"
        elif score >= 0.45:
            return "🟠 Moderate Match - Partial alignment with requirements"
        elif score >= 0.3:
            return "🔴 Weak Match - Limited alignment with requirements"
        else:
            return "⚫ Poor Match - Minimal alignment with requirements"
    
    def _empty_result(self, reason: str) -> Dict[str, any]:
        """
        Return empty result for edge cases.
        
        Args:
            reason: Explanation for empty result
            
        Returns:
            Dictionary with zero similarity
        """
        return {
            "semantic_similarity_score": 0.0,
            "tfidf_similarity": 0.0,
            "embedding_similarity": 0.0,
            "tfidf_weight": self.tfidf_weight,
            "embedding_weight": self.embedding_weight,
            "interpretation": "⚫ Unable to compute similarity",
            "error": reason
        }
    
    def get_configuration(self) -> Dict[str, any]:
        """
        Get current engine configuration.
        
        Returns:
            Dictionary with model settings
        """
        return {
            "tfidf_weight": self.tfidf_weight,
            "embedding_weight": self.embedding_weight,
            "tfidf_params": {
                "max_features": self.tfidf_model.max_features,
                "ngram_range": self.tfidf_model.ngram_range
            },
            "embedding_model": self.embedding_model.get_model_info()
        }
    
    def update_weights(
        self,
        tfidf_weight: float,
        embedding_weight: float
    ) -> None:
        """
        Update similarity weights dynamically.
        
        Args:
            tfidf_weight: New TF-IDF weight
            embedding_weight: New embedding weight
        """
        total = tfidf_weight + embedding_weight
        if total == 0:
            raise ValueError("At least one weight must be > 0")
        
        self.tfidf_weight = tfidf_weight / total
        self.embedding_weight = embedding_weight / total
        
        logger.info(f"Weights updated: TF-IDF={self.tfidf_weight:.2f}, Embedding={self.embedding_weight:.2f}")
