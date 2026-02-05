"""
Unit Tests for Education Extractor
Phase 3: Feature Engineering Tests
"""

import pytest
from app.features.education_extractor import EducationExtractor


@pytest.fixture
def education_extractor():
    """Fixture for education extractor instance"""
    return EducationExtractor()


class TestEducationExtractor:
    
    def test_detect_bachelor_degree(self, education_extractor):
        """Test detection of bachelor's degree"""
        text = "Bachelor of Technology in Computer Science, XYZ University, 2020"
        
        result = education_extractor.extract_education(text)
        
        assert result['highest_degree'] == 'bachelor'
        assert result['degree_level'] == 4
        assert 'computer_science' in result['field_of_study']
    
    def test_detect_master_degree(self, education_extractor):
        """Test detection of master's degree"""
        text = "Master of Science in Data Science, ABC University, 2022"
        
        result = education_extractor.extract_education(text)
        
        assert result['highest_degree'] == 'master'
        assert result['degree_level'] == 5
        assert 'data_science' in result['field_of_study']
    
    def test_detect_phd(self, education_extractor):
        """Test PhD detection"""
        text = "Ph.D. in Computer Science, Stanford University"
        
        result = education_extractor.extract_education(text)
        
        assert result['highest_degree'] == 'phd'
        assert result['degree_level'] == 6
    
    def test_detect_mba(self, education_extractor):
        """Test MBA detection"""
        text = "MBA from Harvard Business School"
        
        result = education_extractor.extract_education(text)
        
        assert result['highest_degree'] == 'mba'
        assert result['degree_level'] == 5
        assert 'business' in result['field_of_study']
    
    def test_education_match_requirement(self, education_extractor):
        """Test education requirement matching"""
        text = "Bachelor's degree in Engineering"
        
        result_met = education_extractor.extract_education(text, required_degree='bachelor')
        result_not_met = education_extractor.extract_education(text, required_degree='master')
        
        assert result_met['education_match'] is True
        assert result_not_met['education_match'] is False
    
    def test_highest_degree_selection(self, education_extractor):
        """Test selection of highest degree when multiple present"""
        text = """
        Bachelor of Science in Computer Science, 2015
        Master of Technology in AI, 2017
        """
        
        result = education_extractor.extract_education(text)
        
        assert result['highest_degree'] == 'master'
        assert result['degree_level'] == 5
    
    def test_check_stem_background(self, education_extractor):
        """Test STEM background detection"""
        stem_fields = ['computer_science', 'engineering', 'mathematics']
        non_stem_fields = ['business', 'arts']
        
        assert education_extractor.check_stem_background(stem_fields) is True
        assert education_extractor.check_stem_background(non_stem_fields) is False
    
    def test_extract_university_names(self, education_extractor):
        """Test university name extraction"""
        text = """
        Bachelor of Technology
        University of California, Berkeley
        Master of Science
        Stanford University
        """
        
        universities = education_extractor.extract_university_names(text)
        
        assert len(universities) >= 1
        assert any('university' in univ.lower() for univ in universities)
    
    def test_no_education_found(self, education_extractor):
        """Test when no education is detected"""
        text = "High school graduate"
        
        result = education_extractor.extract_education(text)
        
        # Should detect high school
        assert result['highest_degree'] in ['high_school', None]
    
    def test_field_of_study_detection(self, education_extractor):
        """Test field of study detection"""
        text = "B.Tech in Information Technology and Engineering"
        
        result = education_extractor.extract_education(text)
        
        assert 'information_technology' in result['field_of_study'] or \
               'engineering' in result['field_of_study']
    
    def test_get_degree_label(self, education_extractor):
        """Test human-readable degree labels"""
        assert education_extractor.get_degree_label('bachelor') == "Bachelor's Degree"
        assert education_extractor.get_degree_label('master') == "Master's Degree"
        assert education_extractor.get_degree_label('phd') == "PhD"
        assert education_extractor.get_degree_label('mba') == "MBA"
