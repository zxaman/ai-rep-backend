"""
Integration Test for Full Resume Analysis Pipeline (Phase 2-5)

Tests the complete flow:
1. Phase 2: Resume Parsing & Text Preprocessing
2. Phase 3: Feature Engineering & Extraction
3. Phase 4: Semantic Similarity Computation
4. Phase 5: Custom Resume Scoring Model

This validates end-to-end system integration and data flow.
"""

import pytest
from io import BytesIO
from fastapi.testclient import TestClient
from app.main import app


class TestFullPipelineIntegration:
    """Test complete resume analysis pipeline from upload to final score"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_resume_text(self):
        """Sample resume text for testing"""
        return """
        John Doe
        Software Engineer
        
        EXPERIENCE:
        Senior Python Developer at Tech Corp (2020-2023)
        - Developed machine learning models using PyTorch and TensorFlow
        - Built scalable REST APIs with FastAPI and Django
        - Implemented CI/CD pipelines with Docker and Kubernetes
        
        SKILLS:
        Python, JavaScript, FastAPI, Django, PyTorch, TensorFlow, Docker, 
        Kubernetes, PostgreSQL, MongoDB, Git, AWS, React, Node.js
        
        PROJECTS:
        1. AI Resume Analyzer - Built NLP-based resume parsing system
        2. Real-time Chat Application - WebSocket server with Redis
        3. E-commerce Platform - Full-stack web application
        
        EDUCATION:
        Bachelor of Science in Computer Science
        University of Technology (2016-2020)
        """
    
    @pytest.fixture
    def sample_job_data(self):
        """Sample job requirements for testing"""
        return {
            "required_skills": [
                "Python", "FastAPI", "Machine Learning", "PyTorch",
                "Docker", "PostgreSQL", "Git", "AWS"
            ],
            "years_of_experience": 3.0,
            "required_degree": "Bachelor"
        }
    
    @pytest.fixture
    def sample_job_description(self):
        """Sample job description text"""
        return """
        We are looking for a skilled Machine Learning Engineer with strong Python expertise.
        The ideal candidate should have experience with:
        - Building ML models using PyTorch or TensorFlow
        - Developing REST APIs with FastAPI or Django
        - Deploying applications with Docker and Kubernetes
        - Working with relational databases like PostgreSQL
        - Cloud platforms (AWS, GCP, or Azure)
        
        Minimum 3 years of professional experience required.
        Bachelor's degree in Computer Science or related field.
        """
    
    def test_health_check(self, client):
        """Test API health check endpoint"""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['status'] == 'healthy'
        assert 'components' in data
        assert 'ml_models' in data
        
        # Verify Phase 5 components are ready
        assert data['components']['scoring_engine'] == 'ready'
    
    def test_root_endpoint_shows_phase_5(self, client):
        """Test root endpoint reflects Phase 5 completion"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['status'] == 'online'
        assert 'Phase 5' in data['phase']
        assert data['version'] == '0.5.0'
    
    def test_step1_parse_resume(self, client, sample_resume_text):
        """Test Step 1: Parse resume and preprocess text"""
        # Create a file-like object
        file_content = sample_resume_text.encode('utf-8')
        files = {'file': ('resume.txt', BytesIO(file_content), 'text/plain')}
        
        response = client.post("/api/parse-resume", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify parsing results
        assert 'cleaned_tokens' in data
        assert 'sections' in data
        assert 'metadata' in data
        
        # Should extract key sections
        assert len(data['sections']) > 0
        assert len(data['cleaned_tokens']) > 0
        
        return data
    
    def test_step2_extract_features(self, client, sample_resume_text, sample_job_data):
        """Test Step 2: Extract features from resume"""
        # First parse the resume
        file_content = sample_resume_text.encode('utf-8')
        files = {'file': ('resume.txt', BytesIO(file_content), 'text/plain')}
        parse_response = client.post("/api/parse-resume", files=files)
        parsed_data = parse_response.json()
        
        # Extract features
        feature_request = {
            "resume_sections": parsed_data['sections'],
            "job_role_data": sample_job_data
        }
        
        response = client.post("/api/extract-features", json=feature_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify feature extraction results
        assert 'skill_match_percent' in data
        assert 'matched_skills' in data
        assert 'missing_skills' in data
        assert 'experience_years' in data
        assert 'project_count' in data
        assert 'normalized_features' in data
        
        # Verify normalized features (0-1 range)
        normalized = data['normalized_features']
        assert 'skill_match' in normalized
        assert 'experience' in normalized
        assert 'project_score' in normalized
        
        for key, value in normalized.items():
            assert 0.0 <= value <= 1.0, f"{key} should be normalized (0-1)"
        
        return data
    
    def test_step3_compute_similarity(self, client, sample_resume_text, sample_job_description):
        """Test Step 3: Compute semantic similarity"""
        similarity_request = {
            "resume_text": sample_resume_text,
            "job_text": sample_job_description,
            "include_details": True
        }
        
        response = client.post("/api/compute-similarity", json=similarity_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify similarity computation results
        assert 'semantic_similarity_score' in data
        assert 'tfidf_similarity' in data
        assert 'embedding_similarity' in data
        
        # Scores should be between 0 and 1
        assert 0.0 <= data['semantic_similarity_score'] <= 1.0
        assert 0.0 <= data['tfidf_similarity'] <= 1.0
        assert 0.0 <= data['embedding_similarity'] <= 1.0
        
        return data
    
    def test_step4_calculate_final_score(self, client):
        """Test Step 4: Calculate final resume score"""
        # Mock normalized features and similarity
        score_request = {
            "normalized_features": {
                "skill_match": 0.75,
                "experience": 0.80,
                "project_score": 0.65
            },
            "semantic_similarity": 0.72,
            "include_breakdown": True
        }
        
        response = client.post("/api/calculate-score", json=score_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify final score results
        assert 'final_score' in data
        assert 'interpretation' in data
        assert 'emoji' in data
        assert 'recommendation' in data
        assert 'weighted_components' in data
        assert 'breakdown' in data
        
        # Score should be between 0 and 100
        assert 0 <= data['final_score'] <= 100
        
        # Verify breakdown structure
        breakdown = data['breakdown']
        assert 'visualizations' in breakdown
        assert 'insights' in breakdown
        
        # Verify visualizations
        viz = breakdown['visualizations']
        assert 'pie_chart' in viz
        assert 'bar_chart' in viz
        assert 'radar_chart' in viz
        assert 'gauge_chart' in viz
        assert 'progress_bars' in viz
        
        # Verify insights
        insights = breakdown['insights']
        assert 'strengths' in insights
        assert 'weaknesses' in insights
        assert 'recommendations' in insights
        
        return data
    
    def test_full_pipeline_end_to_end(self, client, sample_resume_text, 
                                       sample_job_data, sample_job_description):
        """
        Complete end-to-end test of all phases:
        Upload → Parse → Extract → Compute Similarity → Calculate Score
        """
        # Step 1: Parse Resume (Phase 2)
        file_content = sample_resume_text.encode('utf-8')
        files = {'file': ('resume.txt', BytesIO(file_content), 'text/plain')}
        parse_response = client.post("/api/parse-resume", files=files)
        
        assert parse_response.status_code == 200
        parsed_data = parse_response.json()
        
        # Step 2: Extract Features (Phase 3)
        feature_request = {
            "resume_sections": parsed_data['sections'],
            "job_role_data": sample_job_data
        }
        feature_response = client.post("/api/extract-features", json=feature_request)
        
        assert feature_response.status_code == 200
        feature_data = feature_response.json()
        
        # Step 3: Compute Similarity (Phase 4)
        similarity_request = {
            "resume_text": sample_resume_text,
            "job_text": sample_job_description,
            "include_details": True
        }
        similarity_response = client.post("/api/compute-similarity", json=similarity_request)
        
        assert similarity_response.status_code == 200
        similarity_data = similarity_response.json()
        
        # Step 4: Calculate Final Score (Phase 5)
        score_request = {
            "normalized_features": feature_data['normalized_features'],
            "semantic_similarity": similarity_data['semantic_similarity_score'],
            "include_breakdown": True
        }
        score_response = client.post("/api/calculate-score", json=score_request)
        
        assert score_response.status_code == 200
        score_data = score_response.json()
        
        # Verify complete pipeline results
        assert score_data['final_score'] > 0  # Should have a meaningful score
        assert len(score_data['weighted_components']) == 4
        
        # Verify data consistency across phases
        assert feature_data['normalized_features']['skill_match'] >= 0
        assert similarity_data['semantic_similarity_score'] >= 0
        assert score_data['final_score'] >= 0
        
        # Print summary for manual verification
        print("\n" + "="*60)
        print("FULL PIPELINE TEST RESULTS")
        print("="*60)
        print(f"\n1. PARSING (Phase 2):")
        print(f"   - Tokens extracted: {len(parsed_data['cleaned_tokens'])}")
        print(f"   - Sections found: {len(parsed_data['sections'])}")
        
        print(f"\n2. FEATURE EXTRACTION (Phase 3):")
        print(f"   - Skill match: {feature_data['skill_match_percent']}%")
        print(f"   - Matched skills: {len(feature_data['matched_skills'])}")
        print(f"   - Experience: {feature_data['experience_years']} years")
        print(f"   - Projects: {feature_data['project_count']}")
        
        print(f"\n3. SIMILARITY COMPUTATION (Phase 4):")
        print(f"   - Semantic similarity: {similarity_data['semantic_similarity_score']:.2f}")
        print(f"   - TF-IDF similarity: {similarity_data['tfidf_similarity']:.2f}")
        print(f"   - Embedding similarity: {similarity_data['embedding_similarity']:.2f}")
        
        print(f"\n4. FINAL SCORE (Phase 5):")
        print(f"   - Final Score: {score_data['final_score']}/100")
        print(f"   - Interpretation: {score_data['interpretation']} {score_data['emoji']}")
        print(f"   - Recommendation: {score_data['recommendation']}")
        
        print(f"\n   Component Breakdown:")
        for component, value in score_data['weighted_components'].items():
            print(f"     - {component}: {value:.2f}")
        
        print(f"\n   Insights:")
        print(f"     - Strengths: {len(score_data['breakdown']['insights']['strengths'])}")
        print(f"     - Weaknesses: {len(score_data['breakdown']['insights']['weaknesses'])}")
        print(f"     - Recommendations: {len(score_data['breakdown']['insights']['recommendations'])}")
        
        print("\n" + "="*60)
        print("✅ FULL PIPELINE TEST PASSED")
        print("="*60 + "\n")
        
        return {
            'parsed': parsed_data,
            'features': feature_data,
            'similarity': similarity_data,
            'score': score_data
        }
    
    def test_pipeline_with_different_job_roles(self, client, sample_resume_text):
        """Test pipeline with different job role requirements"""
        # Test with high-skill job role
        high_skill_job = {
            "required_skills": [
                "Python", "FastAPI", "PyTorch", "TensorFlow",
                "Docker", "Kubernetes", "AWS", "PostgreSQL",
                "Machine Learning", "Deep Learning", "MLOps"
            ],
            "years_of_experience": 5.0,
            "required_degree": "Master"
        }
        
        # Test with entry-level job role
        entry_level_job = {
            "required_skills": ["Python", "Git", "SQL"],
            "years_of_experience": 1.0,
            "required_degree": "Bachelor"
        }
        
        # Parse resume once
        file_content = sample_resume_text.encode('utf-8')
        files = {'file': ('resume.txt', BytesIO(file_content), 'text/plain')}
        parse_response = client.post("/api/parse-resume", files=files)
        parsed_data = parse_response.json()
        
        # Test with high-skill job
        feature_request_high = {
            "resume_sections": parsed_data['sections'],
            "job_role_data": high_skill_job
        }
        response_high = client.post("/api/extract-features", json=feature_request_high)
        data_high = response_high.json()
        
        # Test with entry-level job
        feature_request_entry = {
            "resume_sections": parsed_data['sections'],
            "job_role_data": entry_level_job
        }
        response_entry = client.post("/api/extract-features", json=feature_request_entry)
        data_entry = response_entry.json()
        
        # Entry-level should have higher skill match percentage
        assert data_entry['skill_match_percent'] >= data_high['skill_match_percent']
        
        # Entry-level should have more matched skills (proportionally)
        entry_match_ratio = len(data_entry['matched_skills']) / len(entry_level_job['required_skills'])
        high_match_ratio = len(data_high['matched_skills']) / len(high_skill_job['required_skills'])
        
        assert entry_match_ratio >= high_match_ratio
    
    def test_pipeline_error_handling(self, client):
        """Test error handling in the pipeline"""
        # Test with invalid file format
        invalid_file = BytesIO(b'\x00\x00\x00\x00')
        files = {'file': ('invalid.bin', invalid_file, 'application/octet-stream')}
        response = client.post("/api/parse-resume", files=files)
        
        # Should handle gracefully (may succeed with basic text extraction or return 400)
        assert response.status_code in [200, 400]
        
        # Test with missing required fields
        invalid_feature_request = {
            "resume_sections": {},
            "job_role_data": {}
        }
        response = client.post("/api/extract-features", json=invalid_feature_request)
        
        # Should return validation error
        assert response.status_code in [400, 422]
        
        # Test with invalid score inputs
        invalid_score_request = {
            "normalized_features": {
                "skill_match": 2.0,  # Invalid: > 1.0
                "experience": 0.5,
                "project_score": 0.5
            },
            "semantic_similarity": 0.5,
            "include_breakdown": True
        }
        response = client.post("/api/calculate-score", json=invalid_score_request)
        
        # Should return error
        assert response.status_code in [400, 422, 500]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
