"""
Text Preprocessor Service
NLP-based text cleaning, tokenization, and lemmatization
Phase 2: NLP Preprocessing Layer
"""

import re
from typing import List
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


class TextPreprocessorService:
    """
    Service for cleaning and preprocessing resume text using spaCy
    Pipeline: lowercase → remove special chars → tokenize → stopword removal → lemmatize
    """
    
    def __init__(self):
        """Initialize spaCy NLP model"""
        try:
            # Load English language model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Disable unnecessary pipeline components for performance
            self.nlp.disable_pipes(["parser", "ner"])
            
            self.stop_words = STOP_WORDS
        
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Please install it with: python -m spacy download en_core_web_sm"
            )
    
    def preprocess(self, text: str) -> List[str]:
        """
        Full preprocessing pipeline
        
        Args:
            text: Raw text extracted from resume
        
        Returns:
            List of cleaned, lemmatized tokens
        """
        # Step 1: Lowercase
        text = text.lower()
        
        # Step 2: Remove special characters (keep alphanumeric and spaces)
        text = self._remove_special_characters(text)
        
        # Step 3: Tokenize with spaCy
        doc = self.nlp(text)
        
        # Step 4: Filter and lemmatize
        cleaned_tokens = []
        for token in doc:
            # Skip stopwords, punctuation, whitespace, single characters
            if (
                token.text not in self.stop_words
                and not token.is_punct
                and not token.is_space
                and len(token.text) > 1
                and token.is_alpha  # Keep only alphabetic tokens
            ):
                # Lemmatize and add to list
                cleaned_tokens.append(token.lemma_)
        
        return cleaned_tokens
    
    def _remove_special_characters(self, text: str) -> str:
        """
        Remove special characters, keeping alphanumeric and basic punctuation
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text with special chars removed
        """
        # Keep letters, numbers, spaces, hyphens, slashes (for skills like C++, .NET)
        pattern = r'[^a-z0-9\s+#\-/\.]'
        cleaned = re.sub(pattern, ' ', text)
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def clean_text_only(self, text: str) -> str:
        """
        Clean text without tokenization (useful for section extraction)
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text as single string
        """
        text = text.lower()
        text = self._remove_special_characters(text)
        return text
    
    def extract_keywords(self, tokens: List[str], top_n: int = 50) -> List[str]:
        """
        Extract most frequent keywords from token list
        
        Args:
            tokens: List of preprocessed tokens
            top_n: Number of top keywords to return
        
        Returns:
            List of top N most frequent tokens
        """
        from collections import Counter
        
        token_counts = Counter(tokens)
        top_keywords = [word for word, count in token_counts.most_common(top_n)]
        
        return top_keywords
