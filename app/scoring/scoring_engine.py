"""
Scoring Engine - Custom Resume Scoring Model
Transparent, explainable scoring logic combining all phases

This is the interpretability layer that calculates final resume scores.

Interview Talking Point:
"I designed the scoring logic myself instead of relying on black-box AI.
The model uses weighted combination of normalized features with full explainability."
"""

from typing import Dict, Optional, Tuple
import json
import os
from pathlib import Path


class ScoringEngine:
    """
    Calculate final resume score from features and similarity.
    
    This engine:
    1. Loads configurable weights from JSON
    2. Validates input features (0-1 normalized)
    3. Applies weighted combination formula
    4. Normalizes to 0-100 scale
    5. Provides score interpretation
    
    Formula:
    final_score = (
        w_skill × skill_match +
        w_similarity × semantic_similarity +
        w_experience × experience +
        w_projects × project_score
    ) × 100
    
    Why this approach?
    - Fully transparent (no black box)
    - Easily configurable (JSON weights)
    - Explainable to stakeholders
    - Domain expert can tune weights
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize scoring engine with configuration.
        
        Args:
            config_path: Path to scoring_config.json (optional)
        """
        if config_path is None:
            # Default to config in same directory
            current_dir = Path(__file__).parent
            config_path = current_dir / "scoring_config.json"
        
        self.config_path = config_path
        self.config = self._load_config()
        self.weights = self.config["weights"]
        self.thresholds = self.config["interpretation_thresholds"]
        
        # Validate weights sum to 1.0
        self._validate_weights()
    
    def _load_config(self) -> Dict:
        """
        Load scoring configuration from JSON file.
        
        Returns:
            Dictionary with scoring configuration
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Scoring config not found at: {self.config_path}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in scoring config: {str(e)}")
    
    def _validate_weights(self) -> None:
        """
        Validate that weights sum to approximately 1.0.
        """
        weight_sum = sum(self.weights.values())
        if not (0.99 <= weight_sum <= 1.01):
            raise ValueError(
                f"Weights must sum to 1.0, got {weight_sum}. "
                f"Current weights: {self.weights}"
            )
    
    def calculate_score(
        self,
        skill_match: float,
        semantic_similarity: float,
        experience: float,
        project_score: float,
        apply_penalties: bool = False,
        apply_bonuses: bool = False
    ) -> Dict[str, any]:
        """
        Calculate final resume score from normalized features.
        
        This is the main scoring method that combines all components.
        
        Args:
            skill_match: Normalized skill match score (0-1)
            semantic_similarity: Semantic similarity score (0-1)
            experience: Normalized experience score (0-1)
            project_score: Normalized project score (0-1)
            apply_penalties: Whether to apply penalty rules (optional)
            apply_bonuses: Whether to apply bonus rules (optional)
        
        Returns:
            {
                "final_score": int (0-100),
                "weighted_components": {...},
                "interpretation": str,
                "label": str,
                "recommendation": str,
                "weights_used": {...}
            }
        """
        # Validate inputs are in 0-1 range
        self._validate_feature(skill_match, "skill_match")
        self._validate_feature(semantic_similarity, "semantic_similarity")
        self._validate_feature(experience, "experience")
        self._validate_feature(project_score, "project_score")
        
        # Calculate weighted components
        skill_component = self.weights["skill_match"] * skill_match
        similarity_component = self.weights["semantic_similarity"] * semantic_similarity
        experience_component = self.weights["experience"] * experience
        project_component = self.weights["projects"] * project_score
        
        # Sum weighted components (0-1 range)
        raw_score = (
            skill_component +
            similarity_component +
            experience_component +
            project_component
        )
        
        # Apply optional penalties and bonuses
        if apply_penalties:
            raw_score = self._apply_penalties(raw_score, skill_match, experience, project_score)
        
        if apply_bonuses:
            raw_score = self._apply_bonuses(raw_score, skill_match, experience)
        
        # Clamp to 0-1 range
        raw_score = max(0.0, min(1.0, raw_score))
        
        # Scale to 0-100
        final_score = int(round(raw_score * 100))
        
        # Get interpretation
        interpretation_data = self._interpret_score(final_score)
        
        return {
            "final_score": final_score,
            "raw_score": float(raw_score),
            "weighted_components": {
                "skill_contribution": float(skill_component * 100),
                "similarity_contribution": float(similarity_component * 100),
                "experience_contribution": float(experience_component * 100),
                "project_contribution": float(project_component * 100)
            },
            "interpretation": interpretation_data["label"],
            "emoji": interpretation_data["emoji"],
            "recommendation": interpretation_data["recommendation"],
            "category": interpretation_data["category"],
            "weights_used": self.weights
        }
    
    def calculate_score_from_features(
        self,
        features: Dict,
        similarity_score: float
    ) -> Dict[str, any]:
        """
        Convenience method to calculate score from feature dict and similarity.
        
        Args:
            features: Dictionary from Phase 3 FeatureBuilder (normalized_features)
            similarity_score: Semantic similarity from Phase 4 (0-1)
        
        Returns:
            Score calculation result dictionary
        """
        # Extract normalized features
        normalized = features.get("normalized_features", {})
        
        return self.calculate_score(
            skill_match=normalized.get("skill_match", 0.0),
            semantic_similarity=similarity_score,
            experience=normalized.get("experience", 0.0),
            project_score=normalized.get("project_score", 0.0)
        )
    
    def _validate_feature(self, value: float, name: str) -> None:
        """
        Validate that a feature is in the correct range.
        
        Args:
            value: Feature value to validate
            name: Feature name for error message
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric, got {type(value)}")
        
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"{name} must be in range [0, 1], got {value}. "
                f"Ensure features are normalized using Phase 3 Normalizer."
            )
    
    def _apply_penalties(
        self,
        score: float,
        skill_match: float,
        experience: float,
        project_score: float
    ) -> float:
        """
        Apply optional penalty rules if enabled in config.
        
        Args:
            score: Current score (0-1)
            skill_match: Skill match score
            experience: Experience score
            project_score: Project score
        
        Returns:
            Adjusted score after penalties
        """
        penalty_rules = self.config.get("penalty_rules", {})
        
        # Missing required skills penalty
        if penalty_rules.get("missing_required_skills", {}).get("enabled", False):
            if skill_match < 1.0:
                penalty_per_skill = penalty_rules["missing_required_skills"]["penalty_per_skill"] / 100
                # Estimate missing skills from match percentage
                penalty = penalty_per_skill * (1.0 - skill_match)
                score -= penalty
        
        # Insufficient experience penalty
        if penalty_rules.get("insufficient_experience", {}).get("enabled", False):
            if experience < 1.0:
                penalty_pct = penalty_rules["insufficient_experience"]["penalty_percentage"] / 100
                score *= (1.0 - penalty_pct)
        
        # No projects penalty
        if penalty_rules.get("no_projects", {}).get("enabled", False):
            if project_score == 0.0:
                penalty = penalty_rules["no_projects"]["penalty"] / 100
                score -= penalty
        
        return score
    
    def _apply_bonuses(
        self,
        score: float,
        skill_match: float,
        experience: float
    ) -> float:
        """
        Apply optional bonus rules if enabled in config.
        
        Args:
            score: Current score (0-1)
            skill_match: Skill match score
            experience: Experience score
        
        Returns:
            Adjusted score after bonuses
        """
        bonus_rules = self.config.get("bonus_rules", {})
        
        # Perfect skill match bonus
        if bonus_rules.get("perfect_skill_match", {}).get("enabled", False):
            if skill_match >= 0.99:
                bonus = bonus_rules["perfect_skill_match"]["bonus"] / 100
                score += bonus
        
        # Exceeds experience bonus
        if bonus_rules.get("exceeds_experience", {}).get("enabled", False):
            # Note: This requires more context about actual vs required experience
            # For now, check if experience is very high (> 0.9)
            if experience >= 0.9:
                bonus_pct = bonus_rules["exceeds_experience"]["bonus_percentage"] / 100
                score += bonus_pct
        
        return score
    
    def _interpret_score(self, score: int) -> Dict[str, str]:
        """
        Interpret score based on threshold ranges.
        
        Args:
            score: Final score (0-100)
        
        Returns:
            Dictionary with interpretation details
        """
        for category, threshold in self.thresholds.items():
            if threshold["min"] <= score <= threshold["max"]:
                return {
                    "category": category,
                    "label": threshold["label"],
                    "emoji": threshold["emoji"],
                    "recommendation": threshold["recommendation"]
                }
        
        # Fallback (should never reach here)
        return {
            "category": "unknown",
            "label": "Unknown",
            "emoji": "❓",
            "recommendation": "Unable to interpret score"
        }
    
    def explain_score(
        self,
        skill_match: float,
        semantic_similarity: float,
        experience: float,
        project_score: float
    ) -> str:
        """
        Generate human-readable explanation of score calculation.
        
        Args:
            skill_match: Normalized skill match score (0-1)
            semantic_similarity: Semantic similarity score (0-1)
            experience: Normalized experience score (0-1)
            project_score: Normalized project score (0-1)
        
        Returns:
            Multi-line string explaining the score
        """
        result = self.calculate_score(
            skill_match, semantic_similarity, experience, project_score
        )
        
        components = result["weighted_components"]
        
        explanation = f"""
═══════════════════════════════════════════════════════
         RESUME SCORE CALCULATION
═══════════════════════════════════════════════════════

🎯 FINAL SCORE: {result['final_score']}/100
{result['emoji']} {result['interpretation']}

───────────────────────────────────────────────────────
📊 SCORE BREAKDOWN
───────────────────────────────────────────────────────

1️⃣ Skill Match: {components['skill_contribution']:.1f}/40 points
   (Weight: {self.weights['skill_match']:.0%})
   Input: {skill_match:.2f} → Contribution: {components['skill_contribution']:.1f}

2️⃣ Semantic Similarity: {components['similarity_contribution']:.1f}/25 points
   (Weight: {self.weights['semantic_similarity']:.0%})
   Input: {semantic_similarity:.2f} → Contribution: {components['similarity_contribution']:.1f}

3️⃣ Experience: {components['experience_contribution']:.1f}/20 points
   (Weight: {self.weights['experience']:.0%})
   Input: {experience:.2f} → Contribution: {components['experience_contribution']:.1f}

4️⃣ Projects: {components['project_contribution']:.1f}/15 points
   (Weight: {self.weights['projects']:.0%})
   Input: {project_score:.2f} → Contribution: {components['project_contribution']:.1f}

───────────────────────────────────────────────────────
💡 INTERPRETATION
───────────────────────────────────────────────────────

{result['recommendation']}

Formula Used:
  final_score = (
    {self.weights['skill_match']} × {skill_match:.2f} +
    {self.weights['semantic_similarity']} × {semantic_similarity:.2f} +
    {self.weights['experience']} × {experience:.2f} +
    {self.weights['projects']} × {project_score:.2f}
  ) × 100 = {result['final_score']}

───────────────────────────────────────────────────────
""".strip()
        
        return explanation
    
    def get_weights(self) -> Dict[str, float]:
        """Get current scoring weights."""
        return self.weights.copy()
    
    def get_thresholds(self) -> Dict:
        """Get score interpretation thresholds."""
        return self.thresholds.copy()
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update scoring weights dynamically.
        
        Args:
            new_weights: Dictionary with new weight values
        """
        # Validate keys
        expected_keys = set(self.weights.keys())
        provided_keys = set(new_weights.keys())
        
        if provided_keys != expected_keys:
            raise ValueError(
                f"Weight keys must match: {expected_keys}. Got: {provided_keys}"
            )
        
        # Update weights
        self.weights.update(new_weights)
        
        # Validate new weights
        self._validate_weights()
