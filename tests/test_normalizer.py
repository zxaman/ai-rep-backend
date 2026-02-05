"""
Unit Tests for Normalizer Utility
Phase 3: Feature Engineering Tests
"""

import pytest
from app.utils.normalizer import Normalizer


class TestNormalizer:
    
    def test_min_max_scale_basic(self):
        """Test basic min-max scaling"""
        # Value in middle of range
        result = Normalizer.min_max_scale(5.0, 0.0, 10.0)
        assert result == 0.5
        
        # Value at minimum
        result = Normalizer.min_max_scale(0.0, 0.0, 10.0)
        assert result == 0.0
        
        # Value at maximum
        result = Normalizer.min_max_scale(10.0, 0.0, 10.0)
        assert result == 1.0
    
    def test_min_max_scale_with_clipping(self):
        """Test min-max scaling with value clipping"""
        # Value below minimum
        result = Normalizer.min_max_scale(-5.0, 0.0, 10.0, clip=True)
        assert result == 0.0
        
        # Value above maximum
        result = Normalizer.min_max_scale(15.0, 0.0, 10.0, clip=True)
        assert result == 1.0
    
    def test_min_max_scale_without_clipping(self):
        """Test min-max scaling without clipping"""
        # Value above maximum
        result = Normalizer.min_max_scale(15.0, 0.0, 10.0, clip=False)
        assert result > 1.0
    
    def test_normalize_skill_match(self):
        """Test skill match normalization"""
        # Perfect match
        result = Normalizer.normalize_skill_match(5, 5)
        assert result == 1.0
        
        # Partial match
        result = Normalizer.normalize_skill_match(3, 5)
        assert result == 0.6
        
        # No match
        result = Normalizer.normalize_skill_match(0, 5)
        assert result == 0.0
        
        # Zero required (edge case)
        result = Normalizer.normalize_skill_match(3, 0)
        assert result == 1.0
    
    def test_normalize_experience_with_requirement(self):
        """Test experience normalization with requirement"""
        # Meets requirement exactly
        result = Normalizer.normalize_experience(5.0, required_years=5.0)
        assert result >= 0.7
        
        # Exceeds requirement
        result = Normalizer.normalize_experience(8.0, required_years=5.0)
        assert result > 0.7
        
        # Below requirement
        result = Normalizer.normalize_experience(3.0, required_years=5.0)
        assert result < 0.7
    
    def test_normalize_experience_without_requirement(self):
        """Test experience normalization without requirement"""
        # Standard scaling
        result = Normalizer.normalize_experience(7.5, min_years=0.0, max_years=15.0)
        assert result == 0.5
    
    def test_normalize_project_count(self):
        """Test project count normalization"""
        # No projects
        result = Normalizer.normalize_project_count(0)
        assert result == 0.0
        
        # Optimal count
        result = Normalizer.normalize_project_count(3, optimal_count=3)
        assert result == 1.0
        
        # Above optimal
        result = Normalizer.normalize_project_count(5, optimal_count=3)
        assert result == 1.0
        
        # Partial
        result = Normalizer.normalize_project_count(2, optimal_count=3)
        assert 0.0 < result < 1.0
    
    def test_normalize_education(self):
        """Test education normalization"""
        # Meets requirement (Bachelor = 4)
        result = Normalizer.normalize_education(4, required_level=4)
        assert result == 1.0
        
        # Exceeds requirement (Master = 5)
        result = Normalizer.normalize_education(5, required_level=4)
        assert result == 1.0
        
        # Below requirement
        result = Normalizer.normalize_education(3, required_level=4)
        assert result < 1.0
    
    def test_clamp(self):
        """Test value clamping"""
        # Value in range
        result = Normalizer.clamp(0.5, 0.0, 1.0)
        assert result == 0.5
        
        # Value below min
        result = Normalizer.clamp(-0.5, 0.0, 1.0)
        assert result == 0.0
        
        # Value above max
        result = Normalizer.clamp(1.5, 0.0, 1.0)
        assert result == 1.0
    
    def test_scale_to_range(self):
        """Test scaling to custom range"""
        # Scale 0-10 to 0-100
        result = Normalizer.scale_to_range(5.0, 0.0, 10.0, 0.0, 100.0)
        assert result == 50.0
        
        # Scale 0-1 to 50-100
        result = Normalizer.scale_to_range(0.5, 0.0, 1.0, 50.0, 100.0)
        assert result == 75.0
    
    def test_normalize_features_dict(self):
        """Test batch feature normalization"""
        features = {
            'skill_match_percent': 0.8,
            'experience_years': 7.5,
            'project_count': 5,
            'degree_level': 5
        }
        
        normalized = Normalizer.normalize_features(features)
        
        # All values should be 0-1
        for key, value in normalized.items():
            assert 0.0 <= value <= 1.0, f"{key} = {value} out of range"
