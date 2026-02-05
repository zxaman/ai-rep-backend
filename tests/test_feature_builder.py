"""
Unit Tests for Feature Builder
Phase 3: Feature Engineering Tests
"""

import pytest
from app.features.feature_builder import FeatureBuilder


@pytest.fixture
def feature_builder():
    """Fixture for feature builder instance"""
    return FeatureBuilder()


@pytest.fixture
def sample_resume_sections():
    """Sample resume sections for testing"""
    return {
        'skills': 'Python, Java, Machine Learning, TensorFlow, AWS, Docker',
        'experience': 'Senior Software Engineer (2018 - Present) with 5 years of experience',
        'projects': '''
        - Built ML sentiment analysis model using Python and TensorFlow
        - Developed e-commerce platform with React and Node.js
        - Created data pipeline for real-time analytics
        ''',
        'education': 'Bachelor of Technology in Computer Science, MIT, 2018'
    }


@pytest.fixture
def sample_job_role_data():
    """Sample job role requirements"""
    return {
        'role_name': 'Machine Learning Engineer',
        'required_skills': ['Python', 'Machine Learning', 'TensorFlow', 'AWS'],
        'min_experience_years': 3.0,
        'required_degree': 'bachelor'
    }


class TestFeatureBuilder:
    
    def test_build_features_complete(self, feature_builder, sample_resume_sections, sample_job_role_data):
        """Test complete feature building process"""
        features = feature_builder.build_features(
            sample_resume_sections,
            sample_job_role_data
        )
        
        # Check all major feature categories present
        assert 'skill_match_percent' in features
        assert 'experience_years' in features
        assert 'project_count' in features
        assert 'highest_degree' in features
        assert 'normalized_features' in features
        assert 'metadata' in features
    
    def test_skill_features(self, feature_builder, sample_resume_sections, sample_job_role_data):
        """Test skill feature extraction"""
        features = feature_builder.build_features(
            sample_resume_sections,
            sample_job_role_data
        )
        
        assert features['skill_match_percent'] > 0.0
        assert len(features['matched_skills']) > 0
        assert features['matched_skill_count'] <= features['required_skill_count']
    
    def test_experience_features(self, feature_builder, sample_resume_sections, sample_job_role_data):
        """Test experience feature extraction"""
        features = feature_builder.build_features(
            sample_resume_sections,
            sample_job_role_data
        )
        
        assert features['experience_years'] >= 3.0
        assert features['experience_match'] is True
        assert features['experience_level'] in [
            'entry_level', 'junior', 'mid_level', 'senior', 'lead', 'expert'
        ]
    
    def test_project_features(self, feature_builder, sample_resume_sections, sample_job_role_data):
        """Test project feature extraction"""
        features = feature_builder.build_features(
            sample_resume_sections,
            sample_job_role_data
        )
        
        assert features['project_count'] >= 2
        assert features['relevant_projects'] >= 1
        assert 0.0 <= features['project_score'] <= 1.0
    
    def test_education_features(self, feature_builder, sample_resume_sections, sample_job_role_data):
        """Test education feature extraction"""
        features = feature_builder.build_features(
            sample_resume_sections,
            sample_job_role_data
        )
        
        assert features['highest_degree'] == 'bachelor'
        assert features['education_match'] is True
        assert features['degree_level'] >= 4
    
    def test_normalized_features(self, feature_builder, sample_resume_sections, sample_job_role_data):
        """Test that all normalized features are in 0-1 range"""
        features = feature_builder.build_features(
            sample_resume_sections,
            sample_job_role_data
        )
        
        normalized = features['normalized_features']
        
        for key, value in normalized.items():
            assert 0.0 <= value <= 1.0, f"Feature {key} = {value} is out of range"
    
    def test_get_feature_summary(self, feature_builder, sample_resume_sections, sample_job_role_data):
        """Test feature summary generation"""
        features = feature_builder.build_features(
            sample_resume_sections,
            sample_job_role_data
        )
        
        summary = feature_builder.get_feature_summary(features)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'Skills' in summary
        assert 'Experience' in summary
        assert 'Projects' in summary
        assert 'Education' in summary
    
    def test_get_feature_vector(self, feature_builder, sample_resume_sections, sample_job_role_data):
        """Test feature vector extraction for ML models"""
        features = feature_builder.build_features(
            sample_resume_sections,
            sample_job_role_data
        )
        
        feature_vector = feature_builder.get_feature_vector(features)
        
        assert isinstance(feature_vector, list)
        assert len(feature_vector) > 0
        assert all(isinstance(val, float) for val in feature_vector)
        assert all(0.0 <= val <= 1.0 for val in feature_vector)
    
    def test_empty_sections(self, feature_builder):
        """Test with empty resume sections"""
        empty_sections = {
            'skills': '',
            'experience': '',
            'projects': '',
            'education': ''
        }
        job_role_data = {
            'role_name': 'Test Role',
            'required_skills': ['Python'],
            'min_experience_years': 2.0,
            'required_degree': 'bachelor'
        }
        
        features = feature_builder.build_features(empty_sections, job_role_data)
        
        assert features['skill_match_percent'] == 0.0
        assert features['experience_years'] == 0.0
        assert features['project_count'] == 0
    
    def test_metadata(self, feature_builder, sample_resume_sections, sample_job_role_data):
        """Test metadata inclusion"""
        features = feature_builder.build_features(
            sample_resume_sections,
            sample_job_role_data
        )
        
        metadata = features['metadata']
        
        assert 'job_role' in metadata
        assert metadata['job_role'] == 'Machine Learning Engineer'
        assert 'extraction_method' in metadata
        assert 'feature_count' in metadata
