"""
Normalizer Utility
Feature normalization and scaling functions
Phase 3: Feature Engineering
"""

from typing import Dict, List, Union, Optional


class Normalizer:
    """
    Utility class for normalizing features to 0-1 range
    
    Supports:
    - Min-Max scaling
    - Custom range scaling
    - Clamping values
    """
    
    @staticmethod
    def min_max_scale(
        value: float,
        min_value: float,
        max_value: float,
        clip: bool = True
    ) -> float:
        """
        Apply min-max normalization to scale value to 0-1 range
        
        Formula: (x - min) / (max - min)
        
        Args:
            value: Value to normalize
            min_value: Minimum value in range
            max_value: Maximum value in range
            clip: Whether to clip result to [0, 1] range
        
        Returns:
            Normalized value in 0-1 range
        """
        if max_value == min_value:
            return 1.0 if value >= max_value else 0.0
        
        scaled = (value - min_value) / (max_value - min_value)
        
        if clip:
            scaled = max(0.0, min(1.0, scaled))
        
        return round(scaled, 4)
    
    @staticmethod
    def normalize_skill_match(matched: int, required: int) -> float:
        """
        Normalize skill match percentage
        
        Args:
            matched: Number of matched skills
            required: Number of required skills
        
        Returns:
            Normalized score 0-1
        """
        if required == 0:
            return 1.0 if matched > 0 else 0.0
        
        percentage = matched / required
        return round(min(1.0, percentage), 2)
    
    @staticmethod
    def normalize_experience(
        years: float,
        required_years: Optional[float] = None,
        min_years: float = 0.0,
        max_years: float = 15.0
    ) -> float:
        """
        Normalize years of experience
        
        Args:
            years: Actual years of experience
            required_years: Required years (if specified, used as threshold)
            min_years: Minimum years for scaling
            max_years: Maximum years for scaling (diminishing returns after this)
        
        Returns:
            Normalized score 0-1
        """
        if required_years is not None:
            # If requirement specified, give bonus for meeting/exceeding it
            if years >= required_years:
                # Met requirement: base 0.7 + bonus for extra years
                extra_years = years - required_years
                bonus = Normalizer.min_max_scale(extra_years, 0, 5) * 0.3
                return round(min(1.0, 0.7 + bonus), 2)
            else:
                # Didn't meet requirement: scale proportionally
                return round(Normalizer.min_max_scale(years, 0, required_years) * 0.7, 2)
        else:
            # No specific requirement: standard scaling
            return Normalizer.min_max_scale(years, min_years, max_years)
    
    @staticmethod
    def normalize_project_count(
        count: int,
        min_count: int = 0,
        optimal_count: int = 3
    ) -> float:
        """
        Normalize project count with diminishing returns
        
        Args:
            count: Number of projects
            min_count: Minimum count (typically 0)
            optimal_count: Optimal number of projects (diminishing returns after)
        
        Returns:
            Normalized score 0-1
        """
        if count <= min_count:
            return 0.0
        
        if count >= optimal_count:
            return 1.0
        
        # Linear scaling up to optimal count
        return Normalizer.min_max_scale(count, min_count, optimal_count)
    
    @staticmethod
    def normalize_education(degree_level: int, required_level: int = 4) -> float:
        """
        Normalize education level
        
        Degree levels:
        1: High School
        2: Diploma
        3: Associate
        4: Bachelor
        5: Master/MBA
        6: PhD
        
        Args:
            degree_level: Candidate's degree level
            required_level: Required degree level (default: 4 = Bachelor)
        
        Returns:
            Normalized score 0-1
        """
        if degree_level >= required_level:
            # Meets or exceeds requirement
            return 1.0
        else:
            # Partial credit for lower degrees
            return Normalizer.min_max_scale(degree_level, 0, required_level)
    
    @staticmethod
    def normalize_features(features: Dict[str, Union[int, float]]) -> Dict[str, float]:
        """
        Normalize multiple features at once
        
        Args:
            features: Dictionary of feature names to raw values
        
        Returns:
            Dictionary with normalized values
        """
        normalized = {}
        
        # Apply appropriate normalization to each known feature
        for key, value in features.items():
            if 'percent' in key or 'score' in key or 'match' in key:
                # Already in 0-1 range, just ensure it's clamped
                normalized[key] = round(max(0.0, min(1.0, float(value))), 4)
            elif 'years' in key:
                # Normalize years (assume max 15 years)
                normalized[key] = Normalizer.normalize_experience(float(value))
            elif 'count' in key:
                # Normalize counts (assume max 10)
                normalized[key] = Normalizer.min_max_scale(float(value), 0, 10)
            elif 'level' in key:
                # Normalize levels (assume max 6)
                normalized[key] = Normalizer.min_max_scale(float(value), 0, 6)
            else:
                # Unknown feature: apply standard 0-1 scaling
                normalized[key] = Normalizer.min_max_scale(float(value), 0, 100)
        
        return normalized
    
    @staticmethod
    def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Clamp value to specified range
        
        Args:
            value: Value to clamp
            min_val: Minimum value
            max_val: Maximum value
        
        Returns:
            Clamped value
        """
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def scale_to_range(
        value: float,
        input_min: float,
        input_max: float,
        output_min: float,
        output_max: float
    ) -> float:
        """
        Scale value from one range to another
        
        Args:
            value: Value to scale
            input_min: Input range minimum
            input_max: Input range maximum
            output_min: Output range minimum
            output_max: Output range maximum
        
        Returns:
            Scaled value
        """
        if input_max == input_min:
            return output_min
        
        # Normalize to 0-1
        normalized = (value - input_min) / (input_max - input_min)
        
        # Scale to output range
        scaled = normalized * (output_max - output_min) + output_min
        
        return round(scaled, 4)
