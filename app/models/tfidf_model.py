"""
TF-IDF Similarity Model
Baseline lexical similarity using Term Frequency-Inverse Document Frequency

This model proves understanding of classic NLP techniques:
- TF-IDF vectorization
- Cosine similarity
- Lexical matching (exact word overlap)

Interview Talking Point:
"I implemented TF-IDF similarity as a baseline model to capture
keyword overlap between resume and job descriptions."
"""

from typing import Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TfidfSimilarityModel:
    """
    Compute lexical similarity using TF-IDF vectors and cosine similarity.
    
    This model:
    1. Fits TF-IDF vectorizer on both texts
    2. Transforms texts to sparse vectors
    3. Computes cosine similarity between vectors
    
    Advantages:
    - Fast computation
    - Explainable (based on word frequency)
    - No pre-trained model needed
    
    Limitations:
    - Only captures exact word matches (not semantic meaning)
    - Misses synonyms (e.g., "Python" vs "Python programming")
    """
    
    def __init__(
        self,
        max_features: int = 500,
        ngram_range: tuple = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95,
        stop_words: str = "english"
    ):
        """
        Initialize TF-IDF model with scikit-learn parameters.
        
        Args:
            max_features: Maximum number of features (terms) to extract
            ngram_range: Range of n-grams (1-word and 2-word phrases)
            min_df: Minimum document frequency for a term
            max_df: Maximum document frequency (filter common words)
            stop_words: Language for stop word removal
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.stop_words = stop_words
        self.vectorizer: Optional[TfidfVectorizer] = None
        
    def compute_similarity(
        self,
        resume_text: str,
        job_text: str
    ) -> Dict[str, any]:
        """
        Compute TF-IDF similarity between resume and job description.
        
        Args:
            resume_text: Cleaned resume text (from Phase 2)
            job_text: Job description or role template text
            
        Returns:
            {
                "tfidf_similarity": float (0-1 range),
                "resume_vector_size": int,
                "job_vector_size": int,
                "shared_terms": List[str],
                "top_resume_terms": List[str],
                "top_job_terms": List[str]
            }
        """
        # Input validation
        if not resume_text or not resume_text.strip():
            return self._empty_result("Empty resume text")
        if not job_text or not job_text.strip():
            return self._empty_result("Empty job description text")
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=self.stop_words,
            lowercase=True,
            strip_accents="unicode"
        )
        
        try:
            # Fit vectorizer on both documents
            corpus = [resume_text, job_text]
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            # Extract vectors
            resume_vector = tfidf_matrix[0:1]
            job_vector = tfidf_matrix[1:2]
            
            # Compute cosine similarity
            similarity_score = cosine_similarity(resume_vector, job_vector)[0][0]
            
            # Get feature names for analysis
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Extract top terms from each document
            resume_dense = resume_vector.toarray()[0]
            job_dense = job_vector.toarray()[0]
            
            top_resume_terms = self._get_top_terms(resume_dense, feature_names, top_n=10)
            top_job_terms = self._get_top_terms(job_dense, feature_names, top_n=10)
            
            # Find shared terms (non-zero in both vectors)
            shared_mask = (resume_dense > 0) & (job_dense > 0)
            shared_terms = [
                feature_names[i]
                for i in np.where(shared_mask)[0]
            ]
            
            return {
                "tfidf_similarity": float(similarity_score),
                "resume_vector_size": int(np.count_nonzero(resume_dense)),
                "job_vector_size": int(np.count_nonzero(job_dense)),
                "shared_terms": shared_terms[:20],  # Limit to 20
                "shared_terms_count": len(shared_terms),
                "top_resume_terms": top_resume_terms,
                "top_job_terms": top_job_terms,
                "vocabulary_size": len(feature_names)
            }
            
        except ValueError as e:
            # Handle edge cases (e.g., all stop words)
            return self._empty_result(f"TF-IDF computation failed: {str(e)}")
    
    def _get_top_terms(
        self,
        vector: np.ndarray,
        feature_names: np.ndarray,
        top_n: int = 10
    ) -> List[str]:
        """
        Extract top N terms from TF-IDF vector by weight.
        
        Args:
            vector: TF-IDF vector (dense array)
            feature_names: Feature names from vectorizer
            top_n: Number of top terms to return
            
        Returns:
            List of top terms sorted by TF-IDF weight
        """
        # Get indices of top terms
        top_indices = np.argsort(vector)[::-1][:top_n]
        
        # Filter out zero weights
        top_indices = [idx for idx in top_indices if vector[idx] > 0]
        
        return [feature_names[idx] for idx in top_indices]
    
    def _empty_result(self, reason: str) -> Dict[str, any]:
        """
        Return empty result for edge cases.
        
        Args:
            reason: Explanation for empty result
            
        Returns:
            Dictionary with zero similarity and empty lists
        """
        return {
            "tfidf_similarity": 0.0,
            "resume_vector_size": 0,
            "job_vector_size": 0,
            "shared_terms": [],
            "shared_terms_count": 0,
            "top_resume_terms": [],
            "top_job_terms": [],
            "vocabulary_size": 0,
            "error": reason
        }
    
    def get_vocabulary(self) -> Optional[Dict[str, int]]:
        """
        Get the vocabulary mapping from the fitted vectorizer.
        
        Returns:
            Dictionary mapping terms to indices, or None if not fitted
        """
        if self.vectorizer is None:
            return None
        return self.vectorizer.vocabulary_
    
    def explain_similarity(
        self,
        resume_text: str,
        job_text: str
    ) -> str:
        """
        Generate human-readable explanation of TF-IDF similarity.
        
        Args:
            resume_text: Resume text
            job_text: Job description text
            
        Returns:
            String explanation of similarity score
        """
        result = self.compute_similarity(resume_text, job_text)
        
        if "error" in result:
            return f"TF-IDF Similarity: Unable to compute ({result['error']})"
        
        similarity = result["tfidf_similarity"]
        shared_count = result["shared_terms_count"]
        
        explanation = f"""
TF-IDF Similarity Analysis
--------------------------
Overall Similarity: {similarity:.2%}

Shared Terms: {shared_count} terms appear in both documents
Vocabulary Size: {result['vocabulary_size']} unique terms total

Top Resume Keywords: {', '.join(result['top_resume_terms'][:5])}
Top Job Keywords: {', '.join(result['top_job_terms'][:5])}

Common Keywords: {', '.join(result['shared_terms'][:10])}

Interpretation:
- {similarity:.0%} lexical overlap between resume and job description
- {"Strong" if similarity > 0.5 else "Moderate" if similarity > 0.3 else "Weak"} keyword match
- This captures exact word matches, not semantic meaning
""".strip()
        
        return explanation
