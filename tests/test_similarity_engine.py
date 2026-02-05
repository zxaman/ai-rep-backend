"""
Unit Tests for Similarity Engine
Integration tests for combined TF-IDF and Embedding similarity
"""

import pytest
from app.models.similarity_engine import SimilarityEngine


@pytest.fixture
def similarity_engine():
    """Create similarity engine with default weights"""
    return SimilarityEngine(
        tfidf_weight=0.3,
        embedding_weight=0.7
    )


@pytest.fixture
def balanced_engine():
    """Create similarity engine with balanced weights"""
    return SimilarityEngine(
        tfidf_weight=0.5,
        embedding_weight=0.5
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


def test_compute_similarity_valid_texts(similarity_engine, sample_resume_text, sample_job_text):
    """Test combined similarity with valid texts"""
    result = similarity_engine.compute_similarity(sample_resume_text, sample_job_text)
    
    # Check main output
    assert "semantic_similarity_score" in result
    assert isinstance(result["semantic_similarity_score"], float)
    assert 0.0 <= result["semantic_similarity_score"] <= 1.0
    
    # Check component scores
    assert "tfidf_similarity" in result
    assert "embedding_similarity" in result
    assert 0.0 <= result["tfidf_similarity"] <= 1.0
    assert 0.0 <= result["embedding_similarity"] <= 1.0
    
    # Check weights
    assert result["tfidf_weight"] == 0.3
    assert result["embedding_weight"] == 0.7
    
    # Check interpretation
    assert "interpretation" in result
    assert isinstance(result["interpretation"], str)


def test_compute_similarity_weighted_combination(similarity_engine, sample_resume_text, sample_job_text):
    """Test that final score is weighted average of components"""
    result = similarity_engine.compute_similarity(sample_resume_text, sample_job_text)
    
    tfidf_score = result["tfidf_similarity"]
    embedding_score = result["embedding_similarity"]
    final_score = result["semantic_similarity_score"]
    
    # Calculate expected weighted average
    expected = 0.3 * tfidf_score + 0.7 * embedding_score
    
    # Final score should match weighted average (within floating point precision)
    assert abs(final_score - expected) < 0.001


def test_compute_similarity_with_details(similarity_engine, sample_resume_text, sample_job_text):
    """Test that detailed results are included when requested"""
    result = similarity_engine.compute_similarity(
        sample_resume_text,
        sample_job_text,
        include_details=True
    )
    
    # Check that detailed results are present
    assert "tfidf_details" in result
    assert "embedding_details" in result
    
    # Check TF-IDF details
    tfidf_details = result["tfidf_details"]
    assert "shared_terms" in tfidf_details
    assert "top_resume_terms" in tfidf_details
    assert "top_job_terms" in tfidf_details
    
    # Check embedding details
    embedding_details = result["embedding_details"]
    assert "embedding_dimension" in embedding_details
    assert "model_name" in embedding_details


def test_compute_similarity_without_details(similarity_engine, sample_resume_text, sample_job_text):
    """Test that detailed results are excluded when not requested"""
    result = similarity_engine.compute_similarity(
        sample_resume_text,
        sample_job_text,
        include_details=False
    )
    
    # Main scores should be present
    assert "semantic_similarity_score" in result
    assert "tfidf_similarity" in result
    assert "embedding_similarity" in result
    
    # Details should not be present
    assert "tfidf_details" not in result
    assert "embedding_details" not in result


def test_compute_similarity_identical_texts(similarity_engine):
    """Test that identical texts give perfect or near-perfect similarity"""
    text = "Python developer with machine learning experience"
    result = similarity_engine.compute_similarity(text, text)
    
    # Should have very high similarity
    assert result["semantic_similarity_score"] >= 0.95


def test_compute_similarity_empty_resume(similarity_engine, sample_job_text):
    """Test handling of empty resume text"""
    result = similarity_engine.compute_similarity("", sample_job_text)
    
    assert result["semantic_similarity_score"] == 0.0
    assert "error" in result


def test_compute_similarity_empty_job(similarity_engine, sample_resume_text):
    """Test handling of empty job description"""
    result = similarity_engine.compute_similarity(sample_resume_text, "")
    
    assert result["semantic_similarity_score"] == 0.0
    assert "error" in result


def test_compute_similarity_unrelated_texts(similarity_engine):
    """Test similarity between completely unrelated texts"""
    resume = "Experienced chef specializing in Italian cuisine and pasta making"
    job = "Looking for software engineer with Java and Spring Boot experience"
    
    result = similarity_engine.compute_similarity(resume, job)
    
    # Unrelated texts should have low similarity
    assert result["semantic_similarity_score"] < 0.4


def test_compute_batch_similarity(similarity_engine, sample_job_text):
    """Test batch similarity computation"""
    resumes = [
        "Python ML engineer with TensorFlow experience",
        "Java developer with Spring Boot",
        "Data scientist with R and statistics"
    ]
    
    results = similarity_engine.compute_batch_similarity(resumes, sample_job_text)
    
    # Should return result for each resume
    assert len(results) == len(resumes)
    
    # Each result should have correct structure
    for i, result in enumerate(results):
        assert result["resume_index"] == i
        assert "semantic_similarity_score" in result
        assert "tfidf_similarity" in result
        assert "embedding_similarity" in result
        assert 0.0 <= result["semantic_similarity_score"] <= 1.0


def test_compute_batch_similarity_ranking(similarity_engine, sample_job_text):
    """Test that batch results can rank resumes by relevance"""
    resumes = [
        "Experienced chef with cooking skills",  # Low match
        "Python developer with ML experience",   # High match
        "Generic office worker",                 # Low match
    ]
    
    results = similarity_engine.compute_batch_similarity(resumes, sample_job_text)
    
    # Resume at index 1 should have highest score
    scores = [r["semantic_similarity_score"] for r in results]
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_explain_similarity(similarity_engine, sample_resume_text, sample_job_text):
    """Test comprehensive explanation generation"""
    explanation = similarity_engine.explain_similarity(sample_resume_text, sample_job_text)
    
    # Check that explanation is generated
    assert isinstance(explanation, str)
    assert len(explanation) > 200
    
    # Check key sections are present
    assert "SEMANTIC SIMILARITY ANALYSIS" in explanation
    assert "OVERALL SIMILARITY" in explanation
    assert "MODEL BREAKDOWN" in explanation
    assert "TF-IDF Similarity" in explanation
    assert "Embedding Similarity" in explanation
    assert "WHAT THIS MEANS" in explanation


def test_get_configuration(similarity_engine):
    """Test configuration retrieval"""
    config = similarity_engine.get_configuration()
    
    assert "tfidf_weight" in config
    assert "embedding_weight" in config
    assert config["tfidf_weight"] == 0.3
    assert config["embedding_weight"] == 0.7
    
    assert "tfidf_params" in config
    assert "embedding_model" in config


def test_update_weights(similarity_engine):
    """Test dynamic weight updates"""
    # Update to balanced weights
    similarity_engine.update_weights(tfidf_weight=0.5, embedding_weight=0.5)
    
    config = similarity_engine.get_configuration()
    assert config["tfidf_weight"] == 0.5
    assert config["embedding_weight"] == 0.5


def test_update_weights_normalization():
    """Test that weights are normalized to sum to 1"""
    engine = SimilarityEngine(tfidf_weight=3, embedding_weight=7)
    
    config = engine.get_configuration()
    
    # Weights should be normalized to sum to 1
    assert config["tfidf_weight"] == 0.3
    assert config["embedding_weight"] == 0.7


def test_update_weights_invalid():
    """Test that invalid weights raise error"""
    engine = SimilarityEngine()
    
    with pytest.raises(ValueError):
        engine.update_weights(tfidf_weight=0, embedding_weight=0)


def test_balanced_weights_effect(balanced_engine, sample_resume_text, sample_job_text):
    """Test that balanced weights give equal importance to both models"""
    result = balanced_engine.compute_similarity(sample_resume_text, sample_job_text)
    
    assert result["tfidf_weight"] == 0.5
    assert result["embedding_weight"] == 0.5
    
    # Final score should be simple average
    expected = 0.5 * result["tfidf_similarity"] + 0.5 * result["embedding_similarity"]
    assert abs(result["semantic_similarity_score"] - expected) < 0.001


def test_interpretation_thresholds(similarity_engine):
    """Test that interpretation changes based on score thresholds"""
    # These will trigger different interpretations based on scores
    test_cases = [
        ("Python ML engineer", "Python ML engineer", "Excellent"),  # High score
        ("Chef cooking Italian food", "Software engineer needed", "Poor"),  # Low score
    ]
    
    for resume, job, expected_keyword in test_cases:
        result = similarity_engine.compute_similarity(resume, job)
        assert expected_keyword in result["interpretation"]


def test_tfidf_only_engine():
    """Test engine with only TF-IDF (zero embedding weight)"""
    engine = SimilarityEngine(tfidf_weight=1.0, embedding_weight=0.0)
    
    resume = "Python developer"
    job = "Python programmer"
    
    result = engine.compute_similarity(resume, job)
    
    # Final score should equal TF-IDF score
    assert abs(result["semantic_similarity_score"] - result["tfidf_similarity"]) < 0.001


def test_embedding_only_engine():
    """Test engine with only embeddings (zero TF-IDF weight)"""
    engine = SimilarityEngine(tfidf_weight=0.0, embedding_weight=1.0)
    
    resume = "Python developer"
    job = "Python programmer"
    
    result = engine.compute_similarity(resume, job)
    
    # Final score should equal embedding score
    assert abs(result["semantic_similarity_score"] - result["embedding_similarity"]) < 0.001


def test_initialization_validation():
    """Test that engine initialization validates parameters"""
    # Valid initialization
    engine = SimilarityEngine(tfidf_weight=0.3, embedding_weight=0.7)
    assert engine is not None
    
    # Invalid weights (out of range)
    with pytest.raises(ValueError):
        SimilarityEngine(tfidf_weight=-0.5, embedding_weight=0.5)
    
    with pytest.raises(ValueError):
        SimilarityEngine(tfidf_weight=1.5, embedding_weight=0.5)
    
    # Both weights zero
    with pytest.raises(ValueError):
        SimilarityEngine(tfidf_weight=0.0, embedding_weight=0.0)
