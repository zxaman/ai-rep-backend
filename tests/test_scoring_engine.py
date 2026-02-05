"""
Unit Tests for Scoring Engine (Phase 5)

Tests transparent scoring formula, weight validation, interpretation logic,
and edge cases for the custom resume scoring model.
"""

import pytest
import json
import os
from pathlib import Path
from app.scoring.scoring_engine import ScoringEngine


class TestScoringEngineInitialization:
    """Test scoring engine initialization and configuration loading"""
    
    def test_init_loads_config(self):
        """Test that scoring engine loads configuration on initialization"""
        engine = ScoringEngine()
        
        assert engine.config is not None
        assert 'weights' in engine.config
        assert 'interpretation_thresholds' in engine.config
        
    def test_init_loads_correct_weights(self):
        """Test that default weights are loaded correctly"""
        engine = ScoringEngine()
        weights = engine.get_weights()
        
        assert weights['skill_match'] == 0.40
        assert weights['semantic_similarity'] == 0.25
        assert weights['experience'] == 0.20
        assert weights['project_score'] == 0.15
        
    def test_weights_sum_to_one(self):
        """Test that all weights sum to 1.0"""
        engine = ScoringEngine()
        weights = engine.get_weights()
        
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.0001  # Allow floating-point tolerance
        
    def test_custom_config_path(self, tmp_path):
        """Test loading custom configuration file"""
        custom_config = {
            "weights": {
                "skill_match": 0.5,
                "semantic_similarity": 0.3,
                "experience": 0.15,
                "project_score": 0.05
            },
            "interpretation_thresholds": {
                "excellent": 90,
                "good": 75,
                "moderate": 60,
                "weak": 45,
                "poor": 0
            }
        }
        
        config_file = tmp_path / "custom_config.json"
        with open(config_file, 'w') as f:
            json.dump(custom_config, f)
        
        engine = ScoringEngine(config_path=str(config_file))
        weights = engine.get_weights()
        
        assert weights['skill_match'] == 0.5
        assert weights['semantic_similarity'] == 0.3


class TestScoreCalculation:
    """Test score calculation logic and formulas"""
    
    def test_calculate_perfect_score(self):
        """Test calculation with perfect scores in all components"""
        engine = ScoringEngine()
        
        result = engine.calculate_score(
            skill_match=1.0,
            semantic_similarity=1.0,
            experience=1.0,
            project_score=1.0
        )
        
        assert result['final_score'] == 100
        assert result['interpretation'] == 'Excellent'
        assert result['emoji'] == '🌟'
        
    def test_calculate_zero_score(self):
        """Test calculation with zero scores in all components"""
        engine = ScoringEngine()
        
        result = engine.calculate_score(
            skill_match=0.0,
            semantic_similarity=0.0,
            experience=0.0,
            project_score=0.0
        )
        
        assert result['final_score'] == 0
        assert result['interpretation'] == 'Poor'
        assert result['emoji'] == '❌'
        
    def test_calculate_weighted_score(self):
        """Test weighted score calculation with mixed components"""
        engine = ScoringEngine()
        
        # Expected: (0.75*0.4 + 0.6*0.25 + 0.8*0.2 + 0.5*0.15) * 100
        # = (0.3 + 0.15 + 0.16 + 0.075) * 100 = 68.5 ≈ 69
        result = engine.calculate_score(
            skill_match=0.75,
            semantic_similarity=0.6,
            experience=0.8,
            project_score=0.5
        )
        
        assert 68 <= result['final_score'] <= 69
        assert result['interpretation'] == 'Moderate'
        
    def test_weighted_components_breakdown(self):
        """Test that weighted components are returned correctly"""
        engine = ScoringEngine()
        
        result = engine.calculate_score(
            skill_match=0.8,
            semantic_similarity=0.7,
            experience=0.6,
            project_score=0.5
        )
        
        components = result['weighted_components']
        
        # skill_match: 0.8 * 0.4 * 100 = 32.0
        assert abs(components['skill_match'] - 32.0) < 0.1
        
        # semantic_similarity: 0.7 * 0.25 * 100 = 17.5
        assert abs(components['semantic_similarity'] - 17.5) < 0.1
        
        # experience: 0.6 * 0.2 * 100 = 12.0
        assert abs(components['experience'] - 12.0) < 0.1
        
        # project_score: 0.5 * 0.15 * 100 = 7.5
        assert abs(components['project_score'] - 7.5) < 0.1
        
    def test_calculate_score_from_features(self):
        """Test convenience method that accepts normalized features dict"""
        engine = ScoringEngine()
        
        features = {
            'skill_match': 0.9,
            'experience': 0.8,
            'project_score': 0.7
        }
        
        result = engine.calculate_score_from_features(
            normalized_features=features,
            semantic_similarity=0.85
        )
        
        assert 'final_score' in result
        assert 'interpretation' in result
        assert result['final_score'] > 0


class TestScoreInterpretation:
    """Test score interpretation thresholds and labels"""
    
    def test_excellent_interpretation(self):
        """Test 'Excellent' interpretation for high scores (85-100)"""
        engine = ScoringEngine()
        
        # Test at 85 (threshold)
        result = engine.calculate_score(0.85, 0.85, 0.85, 0.85)
        assert result['interpretation'] == 'Excellent'
        
        # Test at 95 (mid-range)
        result = engine.calculate_score(0.95, 0.95, 0.95, 0.95)
        assert result['interpretation'] == 'Excellent'
        
    def test_good_interpretation(self):
        """Test 'Good' interpretation for scores (70-84)"""
        engine = ScoringEngine()
        
        # Target score around 75
        result = engine.calculate_score(0.75, 0.75, 0.75, 0.75)
        assert result['interpretation'] == 'Good'
        
    def test_moderate_interpretation(self):
        """Test 'Moderate' interpretation for scores (55-69)"""
        engine = ScoringEngine()
        
        # Target score around 60
        result = engine.calculate_score(0.6, 0.6, 0.6, 0.6)
        assert result['interpretation'] == 'Moderate'
        
    def test_weak_interpretation(self):
        """Test 'Weak' interpretation for scores (40-54)"""
        engine = ScoringEngine()
        
        # Target score around 45
        result = engine.calculate_score(0.45, 0.45, 0.45, 0.45)
        assert result['interpretation'] == 'Weak'
        
    def test_poor_interpretation(self):
        """Test 'Poor' interpretation for low scores (0-39)"""
        engine = ScoringEngine()
        
        # Target score around 20
        result = engine.calculate_score(0.2, 0.2, 0.2, 0.2)
        assert result['interpretation'] == 'Poor'
        
    def test_emoji_mapping(self):
        """Test that correct emojis are assigned to interpretations"""
        engine = ScoringEngine()
        
        # Excellent: 🌟
        result = engine.calculate_score(0.9, 0.9, 0.9, 0.9)
        assert result['emoji'] == '🌟'
        
        # Good: 👍
        result = engine.calculate_score(0.75, 0.75, 0.75, 0.75)
        assert result['emoji'] == '👍'
        
        # Moderate: 👌
        result = engine.calculate_score(0.6, 0.6, 0.6, 0.6)
        assert result['emoji'] == '👌'
        
        # Weak: 👎
        result = engine.calculate_score(0.45, 0.45, 0.45, 0.45)
        assert result['emoji'] == '👎'
        
        # Poor: ❌
        result = engine.calculate_score(0.2, 0.2, 0.2, 0.2)
        assert result['emoji'] == '❌'


class TestInputValidation:
    """Test input validation and error handling"""
    
    def test_negative_inputs_raise_error(self):
        """Test that negative input values raise ValueError"""
        engine = ScoringEngine()
        
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            engine.calculate_score(-0.1, 0.5, 0.5, 0.5)
            
    def test_inputs_above_one_raise_error(self):
        """Test that inputs > 1.0 raise ValueError"""
        engine = ScoringEngine()
        
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            engine.calculate_score(0.5, 1.5, 0.5, 0.5)
            
    def test_invalid_weight_sum_raises_error(self):
        """Test that weights not summing to 1.0 raise ValueError"""
        engine = ScoringEngine()
        
        # Manually corrupt weights to test validation
        engine.config['weights']['skill_match'] = 0.5
        engine.config['weights']['semantic_similarity'] = 0.5
        # Now sum is > 1.0
        
        with pytest.raises(ValueError, match="sum to 1.0"):
            engine._validate_weights()
            
    def test_missing_feature_in_convenience_method(self):
        """Test that missing features in calculate_score_from_features are handled"""
        engine = ScoringEngine()
        
        # Missing 'experience' - should default to 0.0
        features = {
            'skill_match': 0.8,
            'project_score': 0.6
        }
        
        result = engine.calculate_score_from_features(
            normalized_features=features,
            semantic_similarity=0.7
        )
        
        # Should still calculate, treating missing experience as 0.0
        assert result['final_score'] >= 0
        assert result['weighted_components']['experience'] == 0.0


class TestExplainScore:
    """Test human-readable score explanation"""
    
    def test_explain_score_returns_string(self):
        """Test that explain_score returns a string explanation"""
        engine = ScoringEngine()
        
        result = engine.calculate_score(0.8, 0.7, 0.75, 0.65)
        explanation = engine.explain_score(result)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        
    def test_explanation_includes_score(self):
        """Test that explanation includes the final score"""
        engine = ScoringEngine()
        
        result = engine.calculate_score(0.8, 0.7, 0.75, 0.65)
        explanation = engine.explain_score(result)
        
        assert str(result['final_score']) in explanation
        
    def test_explanation_includes_interpretation(self):
        """Test that explanation includes interpretation label"""
        engine = ScoringEngine()
        
        result = engine.calculate_score(0.9, 0.9, 0.9, 0.9)
        explanation = engine.explain_score(result)
        
        assert 'Excellent' in explanation
        
    def test_explanation_includes_components(self):
        """Test that explanation mentions score components"""
        engine = ScoringEngine()
        
        result = engine.calculate_score(0.8, 0.7, 0.75, 0.65)
        explanation = engine.explain_score(result)
        
        # Should mention key components
        assert 'skill' in explanation.lower() or 'skills' in explanation.lower()


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_all_zeros_except_one_component(self):
        """Test scoring when only one component is non-zero"""
        engine = ScoringEngine()
        
        # Only skill_match is 1.0, rest are 0
        result = engine.calculate_score(1.0, 0.0, 0.0, 0.0)
        
        # Should equal skill_match weight * 100 = 40
        assert result['final_score'] == 40
        
    def test_floating_point_precision(self):
        """Test that floating-point calculations are handled correctly"""
        engine = ScoringEngine()
        
        # Use values that might cause floating-point issues
        result = engine.calculate_score(0.33333, 0.66667, 0.11111, 0.88889)
        
        # Should produce a valid score between 0 and 100
        assert 0 <= result['final_score'] <= 100
        assert isinstance(result['final_score'], int)
        
    def test_boundary_threshold_values(self):
        """Test score interpretation at exact threshold boundaries"""
        engine = ScoringEngine()
        
        # Test at exactly 85 (Excellent threshold)
        result = engine.calculate_score(0.85, 0.85, 0.85, 0.85)
        assert result['final_score'] == 85
        assert result['interpretation'] == 'Excellent'
        
        # Test at exactly 70 (Good threshold)
        result = engine.calculate_score(0.7, 0.7, 0.7, 0.7)
        assert result['final_score'] == 70
        assert result['interpretation'] == 'Good'
        
    def test_very_small_positive_values(self):
        """Test with very small but positive input values"""
        engine = ScoringEngine()
        
        result = engine.calculate_score(0.001, 0.001, 0.001, 0.001)
        
        assert result['final_score'] >= 0
        assert result['final_score'] <= 1
        
    def test_recommendation_varies_by_score(self):
        """Test that recommendations change based on score range"""
        engine = ScoringEngine()
        
        excellent_result = engine.calculate_score(0.95, 0.95, 0.95, 0.95)
        poor_result = engine.calculate_score(0.2, 0.2, 0.2, 0.2)
        
        # Recommendations should be different
        assert excellent_result['recommendation'] != poor_result['recommendation']
        
        # Excellent should have positive recommendation
        assert len(excellent_result['recommendation']) > 0
        
        # Poor should have improvement-focused recommendation
        assert len(poor_result['recommendation']) > 0


class TestConfigReload:
    """Test configuration reloading and updates"""
    
    def test_reload_config(self, tmp_path):
        """Test that configuration can be reloaded dynamically"""
        # Create initial config
        config1 = {
            "weights": {
                "skill_match": 0.4,
                "semantic_similarity": 0.25,
                "experience": 0.2,
                "project_score": 0.15
            },
            "interpretation_thresholds": {
                "excellent": 85,
                "good": 70,
                "moderate": 55,
                "weak": 40,
                "poor": 0
            }
        }
        
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config1, f)
        
        engine = ScoringEngine(config_path=str(config_file))
        
        # Get initial score
        result1 = engine.calculate_score(0.8, 0.7, 0.6, 0.5)
        score1 = result1['final_score']
        
        # Modify config with different weights
        config2 = {
            "weights": {
                "skill_match": 0.5,
                "semantic_similarity": 0.3,
                "experience": 0.15,
                "project_score": 0.05
            },
            "interpretation_thresholds": {
                "excellent": 85,
                "good": 70,
                "moderate": 55,
                "weak": 40,
                "poor": 0
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config2, f)
        
        # Reload config
        engine.reload_config()
        
        # Get new score with same inputs
        result2 = engine.calculate_score(0.8, 0.7, 0.6, 0.5)
        score2 = result2['final_score']
        
        # Scores should be different due to weight changes
        assert score1 != score2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
