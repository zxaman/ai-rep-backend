"""
Education Extractor
Detects education level and alignment with job requirements
Phase 3: Feature Engineering
"""

import re
from typing import Dict, Optional, List


class EducationExtractor:
    """
    Rule-based education extraction service
    
    Extracts education information:
    - Degree level detection
    - Field of study identification
    - Education requirement matching
    """
    
    def __init__(self):
        """Initialize education extractor with degree mappings"""
        
        # Degree hierarchy (higher number = higher degree)
        self.degree_hierarchy = {
            'high_school': 1,
            'diploma': 2,
            'associate': 3,
            'bachelor': 4,
            'master': 5,
            'mba': 5,
            'phd': 6,
            'doctorate': 6,
        }
        
        # Degree keyword patterns
        self.degree_patterns = {
            'high_school': [
                r'\bhigh\s+school\b',
                r'\bsecondary\s+school\b',
                r'\b12th\b',
                r'\bhssc\b',
            ],
            'diploma': [
                r'\bdiploma\b',
                r'\bcertificate\b',
                r'\bpolytechnic\b',
            ],
            'associate': [
                r'\bassociate\b',
                r'\ba\.?s\.?\b',
                r'\ba\.?a\.?\b',
            ],
            'bachelor': [
                r'\bbachelor(?:\'?s)?\b',
                r'\bb\.?tech\b',
                r'\bb\.?e\.?\b',
                r'\bb\.?s\.?c?\.?\b',
                r'\bb\.?a\.?\b',
                r'\bundergraduate\b',
                r'\bbs\b',
                r'\bba\b',
            ],
            'master': [
                r'\bmaster(?:\'?s)?\b',
                r'\bm\.?tech\b',
                r'\bm\.?s\.?c?\.?\b',
                r'\bm\.?e\.?\b',
                r'\bm\.?a\.?\b',
                r'\bpostgraduate\b',
                r'\bms\b',
                r'\bma\b',
            ],
            'mba': [
                r'\bmba\b',
                r'\bm\.?b\.?a\.?\b',
                r'\bmaster\s+(?:of|in)\s+business\s+administration\b',
            ],
            'phd': [
                r'\bph\.?d\.?\b',
                r'\bdoctorate\b',
                r'\bdoctoral\b',
            ],
            'doctorate': [
                r'\bdoctorate\b',
                r'\bdoctoral\b',
            ],
        }
        
        # Common field of study keywords
        self.field_patterns = {
            'computer_science': [
                r'computer\s+science', r'cs\b', r'computing',
            ],
            'engineering': [
                r'engineering', r'engineer',
            ],
            'information_technology': [
                r'information\s+technology', r'\bit\b', r'information\s+systems',
            ],
            'data_science': [
                r'data\s+science', r'data\s+analytics',
            ],
            'mathematics': [
                r'mathematics', r'math\b', r'statistics',
            ],
            'business': [
                r'business', r'management', r'commerce',
            ],
            'science': [
                r'science', r'physics', r'chemistry', r'biology',
            ],
        }
    
    def extract_education(
        self,
        education_text: str,
        required_degree: Optional[str] = None
    ) -> Dict:
        """
        Extract education information from resume
        
        Args:
            education_text: Education section from resume
            required_degree: Required degree level for job role
        
        Returns:
            Dict with highest_degree, education_match, field_of_study
        """
        if not education_text:
            return {
                "highest_degree": None,
                "education_match": False,
                "field_of_study": [],
                "degree_level": 0,
            }
        
        # Detect highest degree
        highest_degree = self._detect_highest_degree(education_text)
        degree_level = self.degree_hierarchy.get(highest_degree, 0) if highest_degree else 0
        
        # Detect field of study
        fields = self._detect_field_of_study(education_text)
        
        # Check if education meets requirement
        education_match = True
        if required_degree:
            required_level = self.degree_hierarchy.get(required_degree.lower(), 0)
            education_match = degree_level >= required_level
        
        return {
            "highest_degree": highest_degree,
            "education_match": education_match,
            "field_of_study": fields,
            "degree_level": degree_level,
            "required_degree": required_degree,
        }
    
    def _detect_highest_degree(self, text: str) -> Optional[str]:
        """
        Detect the highest degree level from text
        
        Args:
            text: Education section text
        
        Returns:
            Highest degree level detected or None
        """
        if not text:
            return None
        
        text_lower = text.lower()
        detected_degrees = []
        
        # Check for each degree type
        for degree_type, patterns in self.degree_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected_degrees.append(degree_type)
                    break  # Found this degree type, move to next
        
        if not detected_degrees:
            return None
        
        # Return highest degree based on hierarchy
        highest = max(detected_degrees, key=lambda d: self.degree_hierarchy.get(d, 0))
        return highest
    
    def _detect_field_of_study(self, text: str) -> List[str]:
        """
        Detect fields of study from education text
        
        Args:
            text: Education section text
        
        Returns:
            List of detected fields
        """
        if not text:
            return []
        
        text_lower = text.lower()
        detected_fields = []
        
        for field, patterns in self.field_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected_fields.append(field)
                    break  # Found this field, move to next
        
        return detected_fields
    
    def check_stem_background(self, fields: List[str]) -> bool:
        """
        Check if education includes STEM background
        
        Args:
            fields: List of fields of study
        
        Returns:
            True if STEM background detected
        """
        stem_fields = {
            'computer_science',
            'engineering',
            'information_technology',
            'data_science',
            'mathematics',
            'science',
        }
        
        return any(field in stem_fields for field in fields)
    
    def extract_university_names(self, text: str) -> List[str]:
        """
        Extract university/institution names from education text
        
        Args:
            text: Education section text
        
        Returns:
            List of detected university names
        """
        if not text:
            return []
        
        universities = []
        
        # Common university keywords
        university_patterns = [
            r'university\s+of\s+[\w\s]+',
            r'[\w\s]+\s+university',
            r'[\w\s]+\s+institute\s+of\s+technology',
            r'[\w\s]+\s+college',
        ]
        
        text_lower = text.lower()
        
        for pattern in university_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                univ_name = match.group(0).strip()
                # Basic cleaning
                if len(univ_name) < 100:  # Sanity check
                    universities.append(univ_name.title())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_universities = []
        for univ in universities:
            if univ.lower() not in seen:
                seen.add(univ.lower())
                unique_universities.append(univ)
        
        return unique_universities[:5]  # Return top 5
    
    def get_degree_label(self, degree_type: str) -> str:
        """
        Get human-readable degree label
        
        Args:
            degree_type: Internal degree type
        
        Returns:
            Human-readable label
        """
        labels = {
            'high_school': 'High School',
            'diploma': 'Diploma',
            'associate': 'Associate Degree',
            'bachelor': 'Bachelor\'s Degree',
            'master': 'Master\'s Degree',
            'mba': 'MBA',
            'phd': 'PhD',
            'doctorate': 'Doctorate',
        }
        
        return labels.get(degree_type, degree_type.title())
