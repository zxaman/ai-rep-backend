"""
Score Breakdown Generator
Provides detailed, explainable score breakdowns for UI visualization

This module transforms raw scores into presentation-ready data structures
for charts, progress bars, and detailed explanations.

Interview Talking Point:
"I built a score breakdown system that provides full transparency
into how each component contributes to the final score."
"""

from typing import Dict, List, Optional, Tuple


class ScoreBreakdownGenerator:
    """
    Generate section-wise score breakdowns and visualizations.
    
    This generator:
    1. Takes scoring engine output
    2. Formats data for UI consumption
    3. Generates progress bar data
    4. Creates detailed explanations
    5. Identifies strengths and weaknesses
    
    Output is designed for:
    - Angular charts (Chart.js, D3.js)
    - Progress bars
    - Dashboard cards
    - Detailed reports
    """
    
    def __init__(self, config_weights: Optional[Dict[str, float]] = None):
        """
        Initialize breakdown generator.
        
        Args:
            config_weights: Optional scoring weights (default from scoring_config.json)
        """
        if config_weights is None:
            # Default weights (40/25/20/15)
            config_weights = {
                "skill_match": 0.40,
                "semantic_similarity": 0.25,
                "experience": 0.20,
                "projects": 0.15
            }
        
        self.weights = config_weights
    
    def generate_breakdown(
        self,
        final_score: int,
        weighted_components: Dict[str, float],
        raw_features: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Generate comprehensive score breakdown.
        
        Args:
            final_score: Final resume score (0-100)
            weighted_components: Component contributions from ScoringEngine
            raw_features: Optional raw feature data for detailed analysis
        
        Returns:
            {
                "summary": {...},
                "components": [...],
                "visualizations": {...},
                "insights": {...}
            }
        """
        # Extract component scores
        skill_score = weighted_components["skill_contribution"]
        similarity_score = weighted_components["similarity_contribution"]
        experience_score = weighted_components["experience_contribution"]
        project_score = weighted_components["project_contribution"]
        
        # Generate component details
        components = [
            {
                "name": "Skill Match",
                "score": round(skill_score, 1),
                "max_score": 40,
                "percentage": round((skill_score / 40) * 100, 1),
                "weight": self.weights["skill_match"],
                "emoji": "🎯",
                "color": "#4CAF50",
                "category": "technical"
            },
            {
                "name": "Semantic Similarity",
                "score": round(similarity_score, 1),
                "max_score": 25,
                "percentage": round((similarity_score / 25) * 100, 1),
                "weight": self.weights["semantic_similarity"],
                "emoji": "🔍",
                "color": "#2196F3",
                "category": "alignment"
            },
            {
                "name": "Experience",
                "score": round(experience_score, 1),
                "max_score": 20,
                "percentage": round((experience_score / 20) * 100, 1),
                "weight": self.weights["experience"],
                "emoji": "💼",
                "color": "#FF9800",
                "category": "seniority"
            },
            {
                "name": "Projects",
                "score": round(project_score, 1),
                "max_score": 15,
                "percentage": round((project_score / 15) * 100, 1),
                "weight": self.weights["projects"],
                "emoji": "🚀",
                "color": "#9C27B0",
                "category": "practical"
            }
        ]
        
        # Generate summary
        summary = {
            "total_score": final_score,
            "max_score": 100,
            "percentage": final_score,
            "highest_component": max(components, key=lambda x: x["score"]),
            "lowest_component": min(components, key=lambda x: x["score"]),
            "components_count": len(components)
        }
        
        # Generate visualizations data
        visualizations = self._generate_visualizations(components, final_score)
        
        # Generate insights
        insights = self._generate_insights(components, final_score, raw_features)
        
        return {
            "summary": summary,
            "components": components,
            "visualizations": visualizations,
            "insights": insights
        }
    
    def _generate_visualizations(
        self,
        components: List[Dict],
        final_score: int
    ) -> Dict[str, any]:
        """
        Generate visualization data for charts and graphs.
        
        Args:
            components: List of component dictionaries
            final_score: Final score (0-100)
        
        Returns:
            Dictionary with chart-ready data structures
        """
        # Pie chart data
        pie_chart = {
            "type": "pie",
            "labels": [c["name"] for c in components],
            "values": [c["score"] for c in components],
            "colors": [c["color"] for c in components]
        }
        
        # Bar chart data
        bar_chart = {
            "type": "bar",
            "data": [
                {
                    "label": c["name"],
                    "value": c["score"],
                    "max": c["max_score"],
                    "color": c["color"]
                }
                for c in components
            ]
        }
        
        # Radar/spider chart data
        radar_chart = {
            "type": "radar",
            "labels": [c["name"] for c in components],
            "values": [c["percentage"] for c in components],
            "max": 100
        }
        
        # Progress bars
        progress_bars = [
            {
                "name": c["name"],
                "emoji": c["emoji"],
                "percentage": c["percentage"],
                "score": c["score"],
                "max_score": c["max_score"],
                "color": c["color"],
                "status": self._get_component_status(c["percentage"])
            }
            for c in components
        ]
        
        # Gauge chart for overall score
        gauge_chart = {
            "type": "gauge",
            "value": final_score,
            "max": 100,
            "thresholds": [
                {"value": 40, "color": "#f44336", "label": "Poor"},
                {"value": 55, "color": "#ff9800", "label": "Weak"},
                {"value": 70, "color": "#ffeb3b", "label": "Moderate"},
                {"value": 85, "color": "#8bc34a", "label": "Good"},
                {"value": 100, "color": "#4caf50", "label": "Excellent"}
            ]
        }
        
        return {
            "pie_chart": pie_chart,
            "bar_chart": bar_chart,
            "radar_chart": radar_chart,
            "progress_bars": progress_bars,
            "gauge_chart": gauge_chart
        }
    
    def _generate_insights(
        self,
        components: List[Dict],
        final_score: int,
        raw_features: Optional[Dict]
    ) -> Dict[str, any]:
        """
        Generate insights about strengths, weaknesses, and recommendations.
        
        Args:
            components: List of component dictionaries
            final_score: Final score (0-100)
            raw_features: Optional raw feature data
        
        Returns:
            Dictionary with insights and recommendations
        """
        # Identify strengths (components > 75% of max)
        strengths = [
            c for c in components
            if c["percentage"] >= 75
        ]
        
        # Identify weaknesses (components < 50% of max)
        weaknesses = [
            c for c in components
            if c["percentage"] < 50
        ]
        
        # Identify areas for improvement (50-75%)
        improvements = [
            c for c in components
            if 50 <= c["percentage"] < 75
        ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            strengths, weaknesses, improvements, final_score, raw_features
        )
        
        # Calculate score distribution
        score_distribution = {
            "excellent_range": final_score >= 85,
            "good_range": 70 <= final_score < 85,
            "moderate_range": 55 <= final_score < 70,
            "weak_range": 40 <= final_score < 55,
            "poor_range": final_score < 40
        }
        
        return {
            "strengths": [
                {
                    "component": s["name"],
                    "score": s["score"],
                    "percentage": s["percentage"],
                    "emoji": s["emoji"],
                    "message": f"Strong {s['name'].lower()} performance"
                }
                for s in strengths
            ],
            "weaknesses": [
                {
                    "component": w["name"],
                    "score": w["score"],
                    "percentage": w["percentage"],
                    "emoji": w["emoji"],
                    "message": f"{w['name']} needs improvement"
                }
                for w in weaknesses
            ],
            "improvements": [
                {
                    "component": i["name"],
                    "score": i["score"],
                    "percentage": i["percentage"],
                    "emoji": i["emoji"],
                    "message": f"{i['name']} has room for improvement"
                }
                for i in improvements
            ],
            "recommendations": recommendations,
            "score_distribution": score_distribution
        }
    
    def _generate_recommendations(
        self,
        strengths: List[Dict],
        weaknesses: List[Dict],
        improvements: List[Dict],
        final_score: int,
        raw_features: Optional[Dict]
    ) -> List[str]:
        """
        Generate actionable recommendations based on score analysis.
        
        Args:
            strengths: List of strength components
            weaknesses: List of weakness components
            improvements: List of improvement components
            final_score: Final score
            raw_features: Optional raw feature data
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Overall recommendation
        if final_score >= 85:
            recommendations.append("🟢 Excellent candidate - Strong match for the role")
        elif final_score >= 70:
            recommendations.append("🟡 Good candidate - Proceed with interview")
        elif final_score >= 55:
            recommendations.append("🟠 Moderate fit - Review additional qualifications")
        else:
            recommendations.append("🔴 Below requirements - Consider other candidates")
        
        # Component-specific recommendations
        for weakness in weaknesses:
            name = weakness["name"]
            if name == "Skill Match":
                recommendations.append(
                    "📚 Skills gap detected - Consider candidates with required technical skills"
                )
            elif name == "Semantic Similarity":
                recommendations.append(
                    "🔍 Resume-job alignment is weak - Verify if experience is relevant to role"
                )
            elif name == "Experience":
                recommendations.append(
                    "💼 Experience level below requirements - May need training/mentorship"
                )
            elif name == "Projects":
                recommendations.append(
                    "🚀 Limited project experience - Request portfolio or work samples"
                )
        
        # Positive reinforcement for strengths
        if len(strengths) >= 3:
            recommendations.append(
                "✨ Strong overall profile with multiple strengths"
            )
        
        # Improvement suggestions
        for improvement in improvements:
            name = improvement["name"]
            if name == "Skill Match":
                recommendations.append(
                    "📖 Some skills missing - Ask about willingness to learn during interview"
                )
            elif name == "Experience":
                recommendations.append(
                    "⏱️ Experience slightly below ideal - Consider if other strengths compensate"
                )
        
        return recommendations
    
    def _get_component_status(self, percentage: float) -> str:
        """
        Get status label for a component based on percentage.
        
        Args:
            percentage: Component percentage (0-100)
        
        Returns:
            Status string
        """
        if percentage >= 85:
            return "excellent"
        elif percentage >= 70:
            return "good"
        elif percentage >= 55:
            return "moderate"
        elif percentage >= 40:
            return "weak"
        else:
            return "poor"
    
    def generate_simple_breakdown(
        self,
        final_score: int,
        skill_contribution: float,
        similarity_contribution: float,
        experience_contribution: float,
        project_contribution: float
    ) -> Dict[str, any]:
        """
        Generate simple breakdown for quick display.
        
        Args:
            final_score: Final score (0-100)
            skill_contribution: Skill component score
            similarity_contribution: Similarity component score
            experience_contribution: Experience component score
            project_contribution: Project component score
        
        Returns:
            Simplified breakdown dictionary
        """
        return {
            "total": final_score,
            "skills": round(skill_contribution, 1),
            "similarity": round(similarity_contribution, 1),
            "experience": round(experience_contribution, 1),
            "projects": round(project_contribution, 1),
            "max_scores": {
                "skills": 40,
                "similarity": 25,
                "experience": 20,
                "projects": 15
            }
        }
    
    def format_for_api_response(self, breakdown: Dict) -> Dict:
        """
        Format breakdown data for JSON API response.
        
        Args:
            breakdown: Full breakdown from generate_breakdown()
        
        Returns:
            API-friendly dictionary
        """
        return {
            "summary": breakdown["summary"],
            "components": breakdown["components"],
            "visualizations": {
                "progress_bars": breakdown["visualizations"]["progress_bars"],
                "chart_data": {
                    "pie": breakdown["visualizations"]["pie_chart"],
                    "bar": breakdown["visualizations"]["bar_chart"],
                    "radar": breakdown["visualizations"]["radar_chart"],
                    "gauge": breakdown["visualizations"]["gauge_chart"]
                }
            },
            "insights": {
                "strengths": breakdown["insights"]["strengths"],
                "weaknesses": breakdown["insights"]["weaknesses"],
                "recommendations": breakdown["insights"]["recommendations"]
            }
        }
