"""
Unit Tests for Score Breakdown Generator (Phase 5)

Tests visualization generation, insights generation, and API response formatting
for the transparent resume scoring breakdown.
"""

import pytest
from app.scoring.score_breakdown import ScoreBreakdownGenerator


class TestBreakdownGeneratorInitialization:
    """Test score breakdown generator initialization"""
    
    def test_init_with_default_weights(self):
        """Test initialization with default weights from scoring engine"""
        config_weights = {
            'skill_match': 0.40,
            'semantic_similarity': 0.25,
            'experience': 0.20,
            'project_score': 0.15
        }
        
        generator = ScoreBreakdownGenerator(config_weights=config_weights)
        
        assert generator.config_weights == config_weights
        assert generator.config_weights['skill_match'] == 0.40
        
    def test_init_without_weights(self):
        """Test that generator can initialize without explicit weights"""
        generator = ScoreBreakdownGenerator()
        
        # Should have default or loaded weights
        assert hasattr(generator, 'config_weights')
        assert isinstance(generator.config_weights, dict)


class TestGenerateBreakdown:
    """Test complete breakdown generation"""
    
    def test_generate_breakdown_structure(self):
        """Test that breakdown has required structure"""
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 75,
            'interpretation': 'Good',
            'emoji': '👍',
            'recommendation': 'Strong candidate',
            'weighted_components': {
                'skill_match': 30.0,
                'semantic_similarity': 18.75,
                'experience': 15.0,
                'project_score': 11.25
            }
        }
        
        normalized_features = {
            'skill_match': 0.75,
            'experience': 0.75,
            'project_score': 0.75
        }
        
        breakdown = generator.generate_breakdown(
            score_result=score_result,
            normalized_features=normalized_features,
            semantic_similarity=0.75
        )
        
        # Verify top-level structure
        assert 'visualizations' in breakdown
        assert 'insights' in breakdown
        assert 'metadata' in breakdown
        
    def test_breakdown_includes_all_visualizations(self):
        """Test that breakdown includes all visualization types"""
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 80,
            'interpretation': 'Good',
            'emoji': '👍',
            'recommendation': 'Strong candidate',
            'weighted_components': {
                'skill_match': 32.0,
                'semantic_similarity': 20.0,
                'experience': 16.0,
                'project_score': 12.0
            }
        }
        
        normalized_features = {
            'skill_match': 0.8,
            'experience': 0.8,
            'project_score': 0.8
        }
        
        breakdown = generator.generate_breakdown(
            score_result=score_result,
            normalized_features=normalized_features,
            semantic_similarity=0.8
        )
        
        visualizations = breakdown['visualizations']
        
        # Check all chart types are present
        assert 'pie_chart' in visualizations
        assert 'bar_chart' in visualizations
        assert 'radar_chart' in visualizations
        assert 'gauge_chart' in visualizations
        assert 'progress_bars' in visualizations
        
    def test_breakdown_includes_insights(self):
        """Test that breakdown includes insights section"""
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 65,
            'interpretation': 'Moderate',
            'emoji': '👌',
            'recommendation': 'Decent candidate',
            'weighted_components': {
                'skill_match': 26.0,
                'semantic_similarity': 16.25,
                'experience': 13.0,
                'project_score': 9.75
            }
        }
        
        normalized_features = {
            'skill_match': 0.65,
            'experience': 0.65,
            'project_score': 0.65
        }
        
        breakdown = generator.generate_breakdown(
            score_result=score_result,
            normalized_features=normalized_features,
            semantic_similarity=0.65
        )
        
        insights = breakdown['insights']
        
        # Check insights structure
        assert 'strengths' in insights
        assert 'weaknesses' in insights
        assert 'recommendations' in insights
        
        # Verify lists are non-empty
        assert isinstance(insights['strengths'], list)
        assert isinstance(insights['weaknesses'], list)
        assert isinstance(insights['recommendations'], list)


class TestVisualizationGeneration:
    """Test individual visualization generation methods"""
    
    def test_pie_chart_structure(self):
        """Test pie chart data structure for component weights"""
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 75,
            'interpretation': 'Good',
            'emoji': '👍',
            'recommendation': 'Strong candidate',
            'weighted_components': {
                'skill_match': 30.0,
                'semantic_similarity': 18.75,
                'experience': 15.0,
                'project_score': 11.25
            }
        }
        
        normalized_features = {'skill_match': 0.75, 'experience': 0.75, 'project_score': 0.75}
        
        breakdown = generator.generate_breakdown(score_result, normalized_features, 0.75)
        pie_chart = breakdown['visualizations']['pie_chart']
        
        # Verify pie chart structure
        assert 'labels' in pie_chart
        assert 'data' in pie_chart
        assert 'colors' in pie_chart
        
        # Verify data completeness
        assert len(pie_chart['labels']) == 4  # 4 components
        assert len(pie_chart['data']) == 4
        assert len(pie_chart['colors']) == 4
        
        # Verify labels
        assert 'Skill Match' in pie_chart['labels']
        assert 'Semantic Similarity' in pie_chart['labels']
        
    def test_bar_chart_structure(self):
        """Test bar chart data structure for component comparison"""
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 75,
            'interpretation': 'Good',
            'emoji': '👍',
            'recommendation': 'Strong candidate',
            'weighted_components': {
                'skill_match': 30.0,
                'semantic_similarity': 18.75,
                'experience': 15.0,
                'project_score': 11.25
            }
        }
        
        normalized_features = {'skill_match': 0.75, 'experience': 0.75, 'project_score': 0.75}
        
        breakdown = generator.generate_breakdown(score_result, normalized_features, 0.75)
        bar_chart = breakdown['visualizations']['bar_chart']
        
        # Verify bar chart structure
        assert 'labels' in bar_chart
        assert 'weighted_scores' in bar_chart
        assert 'normalized_values' in bar_chart
        
        # Verify arrays have same length
        assert len(bar_chart['labels']) == len(bar_chart['weighted_scores'])
        assert len(bar_chart['labels']) == len(bar_chart['normalized_values'])
        
    def test_radar_chart_structure(self):
        """Test radar chart data structure for multi-dimensional view"""
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 75,
            'interpretation': 'Good',
            'emoji': '👍',
            'recommendation': 'Strong candidate',
            'weighted_components': {
                'skill_match': 30.0,
                'semantic_similarity': 18.75,
                'experience': 15.0,
                'project_score': 11.25
            }
        }
        
        normalized_features = {'skill_match': 0.75, 'experience': 0.75, 'project_score': 0.75}
        
        breakdown = generator.generate_breakdown(score_result, normalized_features, 0.75)
        radar_chart = breakdown['visualizations']['radar_chart']
        
        # Verify radar chart structure
        assert 'labels' in radar_chart
        assert 'scores' in radar_chart
        assert 'max_value' in radar_chart
        
        # All scores should be between 0 and max_value
        for score in radar_chart['scores']:
            assert 0 <= score <= radar_chart['max_value']
            
    def test_gauge_chart_structure(self):
        """Test gauge chart data structure for overall score display"""
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 75,
            'interpretation': 'Good',
            'emoji': '👍',
            'recommendation': 'Strong candidate',
            'weighted_components': {
                'skill_match': 30.0,
                'semantic_similarity': 18.75,
                'experience': 15.0,
                'project_score': 11.25
            }
        }
        
        normalized_features = {'skill_match': 0.75, 'experience': 0.75, 'project_score': 0.75}
        
        breakdown = generator.generate_breakdown(score_result, normalized_features, 0.75)
        gauge_chart = breakdown['visualizations']['gauge_chart']
        
        # Verify gauge chart structure
        assert 'score' in gauge_chart
        assert 'max_score' in gauge_chart
        assert 'color' in gauge_chart
        assert 'interpretation' in gauge_chart
        
        # Verify values
        assert gauge_chart['score'] == 75
        assert gauge_chart['max_score'] == 100
        assert gauge_chart['interpretation'] == 'Good'
        
    def test_progress_bars_structure(self):
        """Test progress bars data structure for component breakdown"""
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 75,
            'interpretation': 'Good',
            'emoji': '👍',
            'recommendation': 'Strong candidate',
            'weighted_components': {
                'skill_match': 30.0,
                'semantic_similarity': 18.75,
                'experience': 15.0,
                'project_score': 11.25
            }
        }
        
        normalized_features = {'skill_match': 0.75, 'experience': 0.75, 'project_score': 0.75}
        
        breakdown = generator.generate_breakdown(score_result, normalized_features, 0.75)
        progress_bars = breakdown['visualizations']['progress_bars']
        
        # Should be a list of progress bar items
        assert isinstance(progress_bars, list)
        assert len(progress_bars) == 4
        
        # Each item should have required fields
        for item in progress_bars:
            assert 'label' in item
            assert 'percentage' in item
            assert 'color' in item
            
            # Percentage should be between 0 and 100
            assert 0 <= item['percentage'] <= 100


class TestInsightsGeneration:
    """Test insights generation (strengths, weaknesses, recommendations)"""
    
    def test_strengths_identified_for_high_scores(self):
        """Test that strengths are identified for high component scores"""
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 85,
            'interpretation': 'Excellent',
            'emoji': '🌟',
            'recommendation': 'Exceptional candidate',
            'weighted_components': {
                'skill_match': 36.0,  # 0.9 * 40
                'semantic_similarity': 22.5,  # 0.9 * 25
                'experience': 18.0,  # 0.9 * 20
                'project_score': 13.5  # 0.9 * 15
            }
        }
        
        normalized_features = {
            'skill_match': 0.9,
            'experience': 0.9,
            'project_score': 0.9
        }
        
        breakdown = generator.generate_breakdown(score_result, normalized_features, 0.9)
        strengths = breakdown['insights']['strengths']
        
        # Should identify multiple strengths
        assert len(strengths) > 0
        
        # Strengths should be strings
        for strength in strengths:
            assert isinstance(strength, str)
            assert len(strength) > 0
            
    def test_weaknesses_identified_for_low_scores(self):
        """Test that weaknesses are identified for low component scores"""
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 45,
            'interpretation': 'Weak',
            'emoji': '👎',
            'recommendation': 'Needs improvement',
            'weighted_components': {
                'skill_match': 12.0,  # 0.3 * 40
                'semantic_similarity': 7.5,  # 0.3 * 25
                'experience': 6.0,  # 0.3 * 20
                'project_score': 4.5  # 0.3 * 15
            }
        }
        
        normalized_features = {
            'skill_match': 0.3,
            'experience': 0.3,
            'project_score': 0.3
        }
        
        breakdown = generator.generate_breakdown(score_result, normalized_features, 0.3)
        weaknesses = breakdown['insights']['weaknesses']
        
        # Should identify multiple weaknesses
        assert len(weaknesses) > 0
        
        # Weaknesses should be strings
        for weakness in weaknesses:
            assert isinstance(weakness, str)
            assert len(weakness) > 0
            
    def test_recommendations_provided(self):
        """Test that recommendations are generated"""
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 60,
            'interpretation': 'Moderate',
            'emoji': '👌',
            'recommendation': 'Decent candidate with room for growth',
            'weighted_components': {
                'skill_match': 24.0,
                'semantic_similarity': 15.0,
                'experience': 12.0,
                'project_score': 9.0
            }
        }
        
        normalized_features = {
            'skill_match': 0.6,
            'experience': 0.6,
            'project_score': 0.6
        }
        
        breakdown = generator.generate_breakdown(score_result, normalized_features, 0.6)
        recommendations = breakdown['insights']['recommendations']
        
        # Should provide recommendations
        assert len(recommendations) > 0
        
        # Recommendations should be actionable strings
        for recommendation in recommendations:
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0
            
    def test_insights_vary_by_score(self):
        """Test that insights differ based on score levels"""
        generator = ScoreBreakdownGenerator()
        
        # High score
        high_score_result = {
            'final_score': 90,
            'interpretation': 'Excellent',
            'emoji': '🌟',
            'recommendation': 'Exceptional',
            'weighted_components': {
                'skill_match': 36.0,
                'semantic_similarity': 22.5,
                'experience': 18.0,
                'project_score': 13.5
            }
        }
        
        # Low score
        low_score_result = {
            'final_score': 35,
            'interpretation': 'Poor',
            'emoji': '❌',
            'recommendation': 'Significant gaps',
            'weighted_components': {
                'skill_match': 14.0,
                'semantic_similarity': 8.75,
                'experience': 7.0,
                'project_score': 5.25
            }
        }
        
        high_breakdown = generator.generate_breakdown(
            high_score_result,
            {'skill_match': 0.9, 'experience': 0.9, 'project_score': 0.9},
            0.9
        )
        
        low_breakdown = generator.generate_breakdown(
            low_score_result,
            {'skill_match': 0.35, 'experience': 0.35, 'project_score': 0.35},
            0.35
        )
        
        # High score should have more strengths, low score more weaknesses
        assert len(high_breakdown['insights']['strengths']) > len(low_breakdown['insights']['strengths'])
        assert len(low_breakdown['insights']['weaknesses']) > len(high_breakdown['insights']['weaknesses'])


class TestSimpleBreakdown:
    """Test simplified breakdown for quick view"""
    
    def test_generate_simple_breakdown(self):
        """Test simplified breakdown generation"""
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 75,
            'interpretation': 'Good',
            'emoji': '👍',
            'recommendation': 'Strong candidate',
            'weighted_components': {
                'skill_match': 30.0,
                'semantic_similarity': 18.75,
                'experience': 15.0,
                'project_score': 11.25
            }
        }
        
        simple_breakdown = generator.generate_simple_breakdown(score_result)
        
        # Verify structure
        assert 'score' in simple_breakdown
        assert 'interpretation' in simple_breakdown
        assert 'emoji' in simple_breakdown
        assert 'components' in simple_breakdown
        
        # Verify values
        assert simple_breakdown['score'] == 75
        assert simple_breakdown['interpretation'] == 'Good'
        assert len(simple_breakdown['components']) == 4


class TestAPIResponseFormatting:
    """Test API response formatting"""
    
    def test_format_for_api_response(self):
        """Test that breakdown is formatted correctly for API response"""
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 75,
            'interpretation': 'Good',
            'emoji': '👍',
            'recommendation': 'Strong candidate',
            'weighted_components': {
                'skill_match': 30.0,
                'semantic_similarity': 18.75,
                'experience': 15.0,
                'project_score': 11.25
            }
        }
        
        normalized_features = {'skill_match': 0.75, 'experience': 0.75, 'project_score': 0.75}
        
        breakdown = generator.generate_breakdown(score_result, normalized_features, 0.75)
        api_response = generator.format_for_api_response(breakdown)
        
        # Should be a valid dict
        assert isinstance(api_response, dict)
        
        # Should contain all required sections
        assert 'visualizations' in api_response
        assert 'insights' in api_response
        
    def test_api_response_json_serializable(self):
        """Test that API response can be JSON serialized"""
        import json
        
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 75,
            'interpretation': 'Good',
            'emoji': '👍',
            'recommendation': 'Strong candidate',
            'weighted_components': {
                'skill_match': 30.0,
                'semantic_similarity': 18.75,
                'experience': 15.0,
                'project_score': 11.25
            }
        }
        
        normalized_features = {'skill_match': 0.75, 'experience': 0.75, 'project_score': 0.75}
        
        breakdown = generator.generate_breakdown(score_result, normalized_features, 0.75)
        api_response = generator.format_for_api_response(breakdown)
        
        # Should be JSON serializable without errors
        try:
            json_string = json.dumps(api_response)
            assert len(json_string) > 0
        except TypeError:
            pytest.fail("API response is not JSON serializable")


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_perfect_score_breakdown(self):
        """Test breakdown for perfect score (100)"""
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 100,
            'interpretation': 'Excellent',
            'emoji': '🌟',
            'recommendation': 'Perfect match',
            'weighted_components': {
                'skill_match': 40.0,
                'semantic_similarity': 25.0,
                'experience': 20.0,
                'project_score': 15.0
            }
        }
        
        normalized_features = {'skill_match': 1.0, 'experience': 1.0, 'project_score': 1.0}
        
        breakdown = generator.generate_breakdown(score_result, normalized_features, 1.0)
        
        # Should generate valid breakdown even for perfect score
        assert breakdown is not None
        assert 'visualizations' in breakdown
        assert 'insights' in breakdown
        
        # Should have only strengths, no weaknesses
        assert len(breakdown['insights']['strengths']) > 0
        
    def test_zero_score_breakdown(self):
        """Test breakdown for zero score"""
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 0,
            'interpretation': 'Poor',
            'emoji': '❌',
            'recommendation': 'Not a match',
            'weighted_components': {
                'skill_match': 0.0,
                'semantic_similarity': 0.0,
                'experience': 0.0,
                'project_score': 0.0
            }
        }
        
        normalized_features = {'skill_match': 0.0, 'experience': 0.0, 'project_score': 0.0}
        
        breakdown = generator.generate_breakdown(score_result, normalized_features, 0.0)
        
        # Should generate valid breakdown even for zero score
        assert breakdown is not None
        assert 'visualizations' in breakdown
        assert 'insights' in breakdown
        
        # Should have only weaknesses, no strengths
        assert len(breakdown['insights']['weaknesses']) > 0
        
    def test_missing_optional_features(self):
        """Test breakdown with missing optional normalized features"""
        generator = ScoreBreakdownGenerator()
        
        score_result = {
            'final_score': 60,
            'interpretation': 'Moderate',
            'emoji': '👌',
            'recommendation': 'Decent',
            'weighted_components': {
                'skill_match': 24.0,
                'semantic_similarity': 15.0,
                'experience': 12.0,
                'project_score': 9.0
            }
        }
        
        # Only provide some features
        normalized_features = {
            'skill_match': 0.6
        }
        
        # Should handle gracefully
        breakdown = generator.generate_breakdown(score_result, normalized_features, 0.6)
        
        assert breakdown is not None
        assert 'visualizations' in breakdown


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
