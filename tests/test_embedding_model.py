"""
Unit Tests for Embedding Similarity Model
Tests semantic similarity computation using sentence transformers
"""

import pytest
import numpy as np
from app.models.embedding_model import EmbeddingSimilarityModel


@pytest.fixture
def embedding_model():
    """Create embedding model instance for testing"""
    return EmbeddingSimilarityModel(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def sample_resume_text():
    """Sample resume text for testing"""
    return """
    Experienced Python developer with 5 years in machine learning.
    Built ML models using TensorFlow and scikit-learn.
    Strong background in data science, NLP, and deep learning.
    """


@pytest.fixture
def sample_job_text():
    """Sample job description for testing"""
    return """
    Looking for Machine Learning Engineer with Python expertise.
    Must have experience with TensorFlow, PyTorch, or similar frameworks.
    Strong understanding of NLP and deep learning required.
    """


def test_compute_similarity_valid_texts(embedding_model, sample_resume_text, sample_job_text):
    """Test embedding similarity with valid resume and job texts"""
    result = embedding_model.compute_similarity(sample_resume_text, sample_job_text)
    
    # Check that similarity score is computed
    assert "embedding_similarity" in result
    assert isinstance(result["embedding_similarity"], float)
    assert 0.0 <= result["embedding_similarity"] <= 1.0
    
    # Check embedding metadata
    assert "embedding_dimension" in result
    assert result["embedding_dimension"] == 384  # all-MiniLM-L6-v2 dimension
    
    assert "model_name" in result
    assert result["model_name"] == "all-MiniLM-L6-v2"
    
    # Check that embeddings are normalized
    assert result["normalized"] == True
    assert abs(result["resume_embedding_norm"] - 1.0) < 0.01
    assert abs(result["job_embedding_norm"] - 1.0) < 0.01


def test_compute_similarity_identical_texts(embedding_model):
    """Test that identical texts give perfect similarity"""
    text = "Python developer with machine learning experience"
    result = embedding_model.compute_similarity(text, text)
    
    # Identical texts should have similarity close to 1.0
    assert result["embedding_similarity"] >= 0.99


def test_compute_similarity_semantic_match(embedding_model):
    """Test that semantically similar texts (different words) score high"""
    # These texts have similar meaning but different wording
    text1 = "Software engineer with expertise in artificial intelligence"
    text2 = "Developer experienced in machine learning and AI"
    
    result = embedding_model.compute_similarity(text1, text2)
    
    # Should have high similarity despite different words
    assert result["embedding_similarity"] > 0.6
    
    # This is where embeddings outperform TF-IDF!


def test_compute_similarity_empty_resume(embedding_model, sample_job_text):
    """Test handling of empty resume text"""
    result = embedding_model.compute_similarity("", sample_job_text)
    
    assert result["embedding_similarity"] == 0.0
    assert "error" in result
    assert "Empty resume text" in result["error"]


def test_compute_similarity_empty_job(embedding_model, sample_resume_text):
    """Test handling of empty job description"""
    result = embedding_model.compute_similarity(sample_resume_text, "")
    
    assert result["embedding_similarity"] == 0.0
    assert "error" in result
    assert "Empty job description text" in result["error"]


def test_compute_similarity_unrelated_texts(embedding_model):
    """Test similarity between completely unrelated texts"""
    resume = "Experienced chef specializing in Italian cuisine and pasta making"
    job = "Looking for software engineer with Java and Spring Boot experience"
    
    result = embedding_model.compute_similarity(resume, job)
    
    # Unrelated texts should have lower similarity
    assert result["embedding_similarity"] < 0.5


def test_compute_batch_similarity(embedding_model, sample_job_text):
    """Test batch similarity computation for multiple resumes"""
    resumes = [
        "Python developer with ML experience",
        "Java developer with Spring Boot",
        "Data scientist with R and statistics background"
    ]
    
    results = embedding_model.compute_batch_similarity(resumes, sample_job_text)
    
    # Should return result for each resume
    assert len(results) == len(resumes)
    
    # Each result should have similarity score
    for i, result in enumerate(results):
        assert "embedding_similarity" in result
        assert isinstance(result["embedding_similarity"], float)
        assert 0.0 <= result["embedding_similarity"] <= 1.0
        assert result["resume_index"] == i


def test_compute_batch_similarity_empty_list(embedding_model, sample_job_text):
    """Test batch similarity with empty resume list"""
    results = embedding_model.compute_batch_similarity([], sample_job_text)
    assert results == []


def test_compute_batch_similarity_empty_job(embedding_model):
    """Test batch similarity with empty job text"""
    resumes = ["Python developer", "Java developer"]
    results = embedding_model.compute_batch_similarity(resumes, "")
    
    assert len(results) == len(resumes)
    for result in results:
        assert "error" in result


def test_get_embedding(embedding_model):
    """Test raw embedding extraction"""
    text = "Machine learning engineer"
    embedding = embedding_model.get_embedding(text, normalize=True)
    
    # Check embedding is valid numpy array
    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
    
    # Check normalization (L2 norm should be ~1.0)
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 0.01


def test_get_embedding_empty_text(embedding_model):
    """Test embedding extraction with empty text"""
    embedding = embedding_model.get_embedding("")
    assert embedding is None


def test_explain_similarity(embedding_model, sample_resume_text, sample_job_text):
    """Test human-readable explanation generation"""
    explanation = embedding_model.explain_similarity(sample_resume_text, sample_job_text)
    
    # Check that explanation is generated
    assert isinstance(explanation, str)
    assert len(explanation) > 100
    
    # Check key sections are present
    assert "Embedding Similarity Analysis" in explanation
    assert "Overall Similarity" in explanation
    assert "Model:" in explanation
    assert "Embedding Dimension:" in explanation
    assert "Interpretation:" in explanation


def test_get_model_info_before_load(embedding_model):
    """Test model info before model is loaded"""
    info = embedding_model.get_model_info()
    
    assert info["model_name"] == "all-MiniLM-L6-v2"
    assert info["loaded"] == False


def test_get_model_info_after_load(embedding_model):
    """Test model info after model is loaded"""
    # Trigger model load by computing similarity
    embedding_model.compute_similarity("test", "test")
    
    info = embedding_model.get_model_info()
    
    assert info["model_name"] == "all-MiniLM-L6-v2"
    assert info["loaded"] == True
    assert info["embedding_dimension"] == 384
    assert "device" in info
    assert "max_sequence_length" in info


def test_synonym_understanding(embedding_model):
    """Test that model understands synonyms (semantic meaning)"""
    # These use synonyms - embeddings should understand they're similar
    text1 = "Python programmer"
    text2 = "Python developer"
    
    result = embedding_model.compute_similarity(text1, text2)
    
    # Should have very high similarity
    assert result["embedding_similarity"] > 0.85


def test_context_understanding(embedding_model):
    """Test that model understands context"""
    # "ML" in different contexts
    text1 = "Machine learning engineer with ML models experience"
    text2 = "ML engineer building machine learning solutions"
    
    result = embedding_model.compute_similarity(text1, text2)
    
    # Should recognize ML and Machine Learning are same concept
    assert result["embedding_similarity"] > 0.75


def test_normalization_effect(embedding_model):
    """Test effect of embedding normalization"""
    text1 = "Python developer"
    text2 = "Software engineer"
    
    # With normalization
    result_norm = embedding_model.compute_similarity(text1, text2, normalize=True)
    
    # Without normalization
    result_no_norm = embedding_model.compute_similarity(text1, text2, normalize=False)
    
    # Both should give valid scores
    assert 0.0 <= result_norm["embedding_similarity"] <= 1.0
    assert 0.0 <= result_no_norm["embedding_similarity"] <= 1.0
    
    # Normalized should have unit norm
    if "error" not in result_norm:
        assert abs(result_norm["resume_embedding_norm"] - 1.0) < 0.01
