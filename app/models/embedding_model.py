"""
Sentence Embedding Similarity Model
Semantic similarity using pre-trained transformer models

This model proves understanding of modern NLP:
- Sentence transformers (BERT-based)
- Dense vector embeddings
- Semantic meaning capture (not just keywords)

Interview Talking Point:
"I implemented semantic similarity using sentence-transformers with
the all-MiniLM-L6-v2 model to capture meaning beyond exact word matches."
"""

from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EmbeddingSimilarityModel:
    """
    Compute semantic similarity using sentence embeddings.
    
    This model:
    1. Encodes texts using pre-trained transformer model
    2. Generates dense vector embeddings (384 dimensions)
    3. Computes cosine similarity between embeddings
    
    Advantages:
    - Captures semantic meaning (understands synonyms, context)
    - Pre-trained on large corpus (no training needed)
    - Works well with short and long texts
    
    Limitations:
    - Slower than TF-IDF (requires neural network inference)
    - Requires model download (~80MB)
    - Less explainable than keyword matching
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize sentence embedding model.
        
        Args:
            model_name: HuggingFace model identifier
                       (all-MiniLM-L6-v2: 384-dim, 80MB, fast)
            device: Device for inference ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        self.device = device
        self._model: Optional[SentenceTransformer] = None
        
    def _load_model(self) -> SentenceTransformer:
        """
        Lazy-load the sentence transformer model.
        
        Returns:
            Loaded SentenceTransformer model
        """
        if self._model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            logger.info(f"Model loaded successfully (embedding dim: {self._model.get_sentence_embedding_dimension()})")
        return self._model
    
    def compute_similarity(
        self,
        resume_text: str,
        job_text: str,
        normalize: bool = True
    ) -> Dict[str, any]:
        """
        Compute semantic similarity between resume and job description.
        
        Args:
            resume_text: Cleaned resume text (from Phase 2)
            job_text: Job description or role template text
            normalize: Whether to normalize embeddings (recommended)
            
        Returns:
            {
                "embedding_similarity": float (0-1 range),
                "resume_embedding_norm": float,
                "job_embedding_norm": float,
                "embedding_dimension": int,
                "model_name": str
            }
        """
        # Input validation
        if not resume_text or not resume_text.strip():
            return self._empty_result("Empty resume text")
        if not job_text or not job_text.strip():
            return self._empty_result("Empty job description text")
        
        try:
            # Load model
            model = self._load_model()
            
            # Encode texts to embeddings
            resume_embedding = model.encode(
                resume_text,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            
            job_embedding = model.encode(
                job_text,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            
            # Compute cosine similarity
            similarity_score = cosine_similarity(
                resume_embedding.reshape(1, -1),
                job_embedding.reshape(1, -1)
            )[0][0]
            
            # Calculate embedding norms (useful for debugging)
            resume_norm = float(np.linalg.norm(resume_embedding))
            job_norm = float(np.linalg.norm(job_embedding))
            
            return {
                "embedding_similarity": float(similarity_score),
                "resume_embedding_norm": resume_norm,
                "job_embedding_norm": job_norm,
                "embedding_dimension": len(resume_embedding),
                "model_name": self.model_name,
                "normalized": normalize
            }
            
        except Exception as e:
            logger.error(f"Embedding similarity computation failed: {str(e)}")
            return self._empty_result(f"Embedding computation failed: {str(e)}")
    
    def compute_batch_similarity(
        self,
        resume_texts: List[str],
        job_text: str,
        normalize: bool = True
    ) -> List[Dict[str, any]]:
        """
        Compute similarity for multiple resumes against one job description.
        More efficient than calling compute_similarity multiple times.
        
        Args:
            resume_texts: List of resume texts
            job_text: Job description text
            normalize: Whether to normalize embeddings
            
        Returns:
            List of similarity results for each resume
        """
        if not resume_texts:
            return []
        
        if not job_text or not job_text.strip():
            return [self._empty_result("Empty job description text") for _ in resume_texts]
        
        try:
            model = self._load_model()
            
            # Encode all resumes in batch (more efficient)
            resume_embeddings = model.encode(
                resume_texts,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            
            # Encode job description once
            job_embedding = model.encode(
                job_text,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            
            # Compute similarity for each resume
            results = []
            for i, resume_embedding in enumerate(resume_embeddings):
                similarity_score = cosine_similarity(
                    resume_embedding.reshape(1, -1),
                    job_embedding.reshape(1, -1)
                )[0][0]
                
                results.append({
                    "embedding_similarity": float(similarity_score),
                    "resume_index": i,
                    "model_name": self.model_name
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch embedding similarity failed: {str(e)}")
            return [self._empty_result(str(e)) for _ in resume_texts]
    
    def get_embedding(
        self,
        text: str,
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Get raw embedding vector for a text.
        Useful for caching or custom similarity computations.
        
        Args:
            text: Input text
            normalize: Whether to normalize embedding
            
        Returns:
            Numpy array of embedding vector, or None if failed
        """
        if not text or not text.strip():
            return None
        
        try:
            model = self._load_model()
            embedding = model.encode(
                text,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            return embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {str(e)}")
            return None
    
    def _empty_result(self, reason: str) -> Dict[str, any]:
        """
        Return empty result for edge cases.
        
        Args:
            reason: Explanation for empty result
            
        Returns:
            Dictionary with zero similarity
        """
        return {
            "embedding_similarity": 0.0,
            "resume_embedding_norm": 0.0,
            "job_embedding_norm": 0.0,
            "embedding_dimension": 0,
            "model_name": self.model_name,
            "normalized": False,
            "error": reason
        }
    
    def explain_similarity(
        self,
        resume_text: str,
        job_text: str
    ) -> str:
        """
        Generate human-readable explanation of embedding similarity.
        
        Args:
            resume_text: Resume text
            job_text: Job description text
            
        Returns:
            String explanation of similarity score
        """
        result = self.compute_similarity(resume_text, job_text)
        
        if "error" in result:
            return f"Embedding Similarity: Unable to compute ({result['error']})"
        
        similarity = result["embedding_similarity"]
        
        # Interpret similarity score
        if similarity >= 0.8:
            interpretation = "Excellent semantic match"
        elif similarity >= 0.65:
            interpretation = "Good semantic match"
        elif similarity >= 0.5:
            interpretation = "Moderate semantic match"
        elif similarity >= 0.35:
            interpretation = "Weak semantic match"
        else:
            interpretation = "Poor semantic match"
        
        explanation = f"""
Embedding Similarity Analysis
------------------------------
Overall Similarity: {similarity:.2%}

Model: {result['model_name']}
Embedding Dimension: {result['embedding_dimension']}

Resume Embedding Norm: {result['resume_embedding_norm']:.4f}
Job Embedding Norm: {result['job_embedding_norm']:.4f}

Interpretation: {interpretation}

Technical Details:
- Uses pre-trained transformer model (BERT-based)
- Captures semantic meaning, not just keywords
- Cosine similarity of dense vector embeddings
- Scores typically range from 0.3 (poor) to 0.9 (excellent)

What this means:
- {similarity:.0%} semantic alignment between resume and job role
- This model understands synonyms and context
- Higher score = candidate's experience aligns with job requirements
""".strip()
        
        return explanation
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        if self._model is None:
            return {
                "model_name": self.model_name,
                "loaded": False,
                "device": self.device
            }
        
        return {
            "model_name": self.model_name,
            "loaded": True,
            "device": str(self._model.device),
            "embedding_dimension": self._model.get_sentence_embedding_dimension(),
            "max_sequence_length": self._model.max_seq_length
        }
