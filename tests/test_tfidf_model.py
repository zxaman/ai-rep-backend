"""
Unit Tests for TF-IDF Similarity Model
Tests lexical similarity computation, edge cases, and explainability
"""

import pytest
from app.models.tfidf_model import TfidfSimilarityModel


@pytest.fixture
def tfidf_model():
    """Create TF-IDF model instance for testing"""
    return TfidfSimilarityModel(
        max_features=100,
        ngram_range=(1, 2),
        min_df=1
    )


@pytest.fixture
def sample_resume_text():
    """Sample resume text for testing"""
    return """
    Experienced Python developer with 5 years in machine learning.
    Built ML models using TensorFlow and scikit-learn.
    Strong background in data science, NLP, and deep learning.
    Projects: sentiment analysis, recommendation systems, computer vision.
    Education: Master's in Computer Science.
    """


@pytest.fixture
def sample_job_text():
    """Sample job description for testing"""
    return """
    Looking for Machine Learning Engineer with Python expertise.
    Must have experience with TensorFlow, PyTorch, or similar frameworks.
    Strong understanding of NLP and deep learning required.
    Bachelor's degree in Computer Science or related field.
    """


def test_compute_similarity_valid_texts(tfidf_model, sample_resume_text, sample_job_text):
    """Test TF-IDF similarity with valid resume and job texts"""
    result = tfidf_model.compute_similarity(sample_resume_text, sample_job_text)
    
    # Check that similarity score is computed
    assert "tfidf_similarity" in result
    assert isinstance(result["tfidf_similarity"], float)
    assert 0.0 <= result["tfidf_similarity"] <= 1.0
    
    # Check that vectors are non-empty
    assert result["resume_vector_size"] > 0
    assert result["job_vector_size"] > 0
    
    # Check that shared terms exist
    assert "shared_terms" in result
    assert isinstance(result["shared_terms"], list)
    assert result["shared_terms_count"] > 0
    
    # Check that top terms are extracted
    assert "top_resume_terms" in result
    assert "top_job_terms" in result
    assert len(result["top_resume_terms"]) > 0
    assert len(result["top_job_terms"]) > 0


def test_compute_similarity_identical_texts(tfidf_model):
    """Test that identical texts give perfect similarity"""
    text = "Python developer with machine learning experience"
    result = tfidf_model.compute_similarity(text, text)
    
    # Identical texts should have similarity close to 1.0
    assert result["tfidf_similarity"] >= 0.99


def test_compute_similarity_empty_resume(tfidf_model, sample_job_text):
    """Test handling of empty resume text"""
    result = tfidf_model.compute_similarity("", sample_job_text)
    
    assert result["tfidf_similarity"] == 0.0
    assert "error" in result
    assert "Empty resume text" in result["error"]


def test_compute_similarity_empty_job(tfidf_model, sample_resume_text):
    """Test handling of empty job description"""
    result = tfidf_model.compute_similarity(sample_resume_text, "")
    
    assert result["tfidf_similarity"] == 0.0
    assert "error" in result
    assert "Empty job description text" in result["error"]


def test_compute_similarity_unrelated_texts(tfidf_model):
    """Test similarity between completely unrelated texts"""
    resume = "Experienced chef specializing in Italian cuisine and pasta making"
    job = "Looking for software engineer with Java and Spring Boot experience"
    
    result = tfidf_model.compute_similarity(resume, job)
    
    # Unrelated texts should have low similarity
    assert result["tfidf_similarity"] < 0.3
    assert result["shared_terms_count"] < 5


def test_compute_similarity_partial_overlap(tfidf_model):
    """Test similarity with partial keyword overlap"""
    resume = "Python developer with experience in web development using Django and Flask"
    job = "Python developer needed for data science projects with pandas and numpy"
    
    result = tfidf_model.compute_similarity(resume, job)
    
    # Should have moderate similarity (Python is shared)
    assert 0.2 <= result["tfidf_similarity"] <= 0.8
    assert "python" in [term.lower() for term in result["shared_terms"]]


def test_get_top_terms(tfidf_model, sample_resume_text, sample_job_text):
    """Test that top terms are correctly extracted"""
    result = tfidf_model.compute_similarity(sample_resume_text, sample_job_text)
    
    top_resume = result["top_resume_terms"]
    top_job = result["top_job_terms"]
    
    # Should extract meaningful terms
    assert len(top_resume) > 0
    assert len(top_job) > 0
    
    # Terms should be strings
    assert all(isinstance(term, str) for term in top_resume)
    assert all(isinstance(term, str) for term in top_job)


def test_shared_terms_detection(tfidf_model):
    """Test detection of shared terms between texts"""
    resume = "Python machine learning data science"
    job = "Python data science experience required"
    
    result = tfidf_model.compute_similarity(resume, job)
    
    shared = [term.lower() for term in result["shared_terms"]]
    
    # These terms should be shared
    assert "python" in shared
    assert "data" in shared or "data science" in shared


def test_explain_similarity(tfidf_model, sample_resume_text, sample_job_text):
    """Test human-readable explanation generation"""
    explanation = tfidf_model.explain_similarity(sample_resume_text, sample_job_text)
    
    # Check that explanation is generated
    assert isinstance(explanation, str)
    assert len(explanation) > 100
    
    # Check key sections are present
    assert "TF-IDF Similarity Analysis" in explanation
    assert "Overall Similarity" in explanation
    assert "Shared Terms" in explanation
    assert "Top Resume Keywords" in explanation
    assert "Top Job Keywords" in explanation


def test_vocabulary_extraction(tfidf_model, sample_resume_text, sample_job_text):
    """Test vocabulary extraction from fitted model"""
    # Initially no vocabulary
    assert tfidf_model.get_vocabulary() is None
    
    # After fitting, vocabulary should exist
    result = tfidf_model.compute_similarity(sample_resume_text, sample_job_text)
    vocab = tfidf_model.get_vocabulary()
    
    assert vocab is not None
    assert isinstance(vocab, dict)
    assert len(vocab) > 0
    
    # Vocabulary should map terms to indices
    for term, idx in vocab.items():
        assert isinstance(term, str)
        assert isinstance(idx, int)


def test_ngram_extraction(tfidf_model):
    """Test that n-grams (bigrams) are extracted"""
    resume = "machine learning engineer with deep learning experience"
    job = "machine learning position available"
    
    result = tfidf_model.compute_similarity(resume, job)
    
    # Should capture bigrams like "machine learning"
    vocab = tfidf_model.get_vocabulary()
    terms = list(vocab.keys()) if vocab else []
    
    # Check for presence of bigrams (2-word phrases)
    bigrams = [term for term in terms if ' ' in term]
    assert len(bigrams) > 0, "Should extract bigrams with ngram_range=(1,2)"


def test_case_insensitive_matching(tfidf_model):
    """Test that matching is case-insensitive"""
    resume = "Python Developer"
    job = "python developer"
    
    result = tfidf_model.compute_similarity(resume, job)
    
    # Should have perfect or near-perfect similarity despite case differences
    assert result["tfidf_similarity"] >= 0.95
