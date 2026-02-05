"""
Unit Tests for Project Extractor
Phase 3: Feature Engineering Tests
"""

import pytest
from app.features.project_extractor import ProjectExtractor


@pytest.fixture
def project_extractor():
    """Fixture for project extractor instance"""
    return ProjectExtractor()


class TestProjectExtractor:
    
    def test_count_projects_with_bullets(self, project_extractor):
        """Test project counting with bullet points"""
        text = """
        - Built e-commerce platform using React and Node.js
        - Developed machine learning model for sentiment analysis
        - Created data pipeline for ETL processing
        """
        
        result = project_extractor.extract_projects(text)
        
        assert result['project_count'] >= 3
        assert result['has_projects'] is True
    
    def test_count_projects_with_indicators(self, project_extractor):
        """Test project counting based on action verbs"""
        text = """
        Built a real-time chat application
        Developed automated testing framework
        Implemented microservices architecture
        """
        
        result = project_extractor.extract_projects(text)
        
        assert result['project_count'] >= 2
    
    def test_relevant_projects_matching(self, project_extractor):
        """Test relevant project detection"""
        text = """
        - Machine Learning project: Built sentiment analysis model using Python and TensorFlow
        - Web Development: Created full-stack application with React
        - Data Analysis: Analyzed customer data using SQL
        """
        job_keywords = ["Python", "Machine Learning", "TensorFlow", "Data"]
        
        result = project_extractor.extract_projects(text, job_keywords)
        
        assert result['relevant_projects'] >= 2
        assert result['relevant_projects'] <= result['project_count']
    
    def test_project_score_calculation(self, project_extractor):
        """Test project score calculation"""
        # Test with 3 relevant projects
        text = """
        - Python ML project
        - Python data pipeline
        - Python automation
        """
        keywords = ["Python"]
        
        result = project_extractor.extract_projects(text, keywords)
        
        assert 0.0 <= result['project_score'] <= 1.0
        assert result['project_score'] > 0.5  # Should score well
    
    def test_no_projects(self, project_extractor):
        """Test with no projects"""
        text = ""
        
        result = project_extractor.extract_projects(text)
        
        assert result['project_count'] == 0
        assert result['relevant_projects'] == 0
        assert result['project_score'] == 0.0
        assert result['has_projects'] is False
    
    def test_identify_project_domains(self, project_extractor):
        """Test project domain identification"""
        text = """
        Built web application with React and Node.js
        Deployed ML model on AWS cloud
        Created mobile app using React Native
        """
        
        domains = project_extractor.identify_project_domains(text)
        
        assert 'web' in domains
        assert 'cloud' in domains
        assert 'mobile' in domains
    
    def test_project_with_long_descriptions(self, project_extractor):
        """Test with detailed project descriptions"""
        text = """
        E-Commerce Platform
        Built a full-stack e-commerce platform using MERN stack.
        Implemented user authentication, shopping cart functionality,
        and integrated payment gateway. Deployed on AWS EC2.
        
        Machine Learning Model
        Developed a sentiment analysis model using Python and TensorFlow.
        Achieved 92% accuracy on test data. Deployed as REST API using Flask.
        """
        
        result = project_extractor.extract_projects(text)
        
        assert result['project_count'] >= 2
