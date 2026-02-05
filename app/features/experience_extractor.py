"""
Experience Extractor
Estimates years of professional experience from resume text
Phase 3: Feature Engineering
"""

import re
from typing import Dict, Optional, List, Tuple
from datetime import datetime


class ExperienceExtractor:
    """
    Rule-based experience extraction service
    
    Extracts years of experience using:
    - Explicit "X years" patterns
    - Date range calculations (2019-2023)
    - Month-year formats (Jan 2020 - Dec 2023)
    - Present/Current employment handling
    """
    
    def __init__(self):
        """Initialize experience extractor with patterns"""
        
        # Month name mappings
        self.month_map = {
            'jan': 1, 'january': 1,
            'feb': 2, 'february': 2,
            'mar': 3, 'march': 3,
            'apr': 4, 'april': 4,
            'may': 5,
            'jun': 6, 'june': 6,
            'jul': 7, 'july': 7,
            'aug': 8, 'august': 8,
            'sep': 9, 'sept': 9, 'september': 9,
            'oct': 10, 'october': 10,
            'nov': 11, 'november': 11,
            'dec': 12, 'december': 12,
        }
    
    def extract_experience(
        self,
        experience_text: str,
        required_years: Optional[float] = None
    ) -> Dict:
        """
        Extract years of experience from resume experience section
        
        Args:
            experience_text: Experience section from resume
            required_years: Minimum years required for job role (optional)
        
        Returns:
            Dict with experience_years, experience_match, calculation_method
        """
        if not experience_text:
            return {
                "experience_years": 0.0,
                "experience_match": False,
                "calculation_method": "no_data",
                "date_ranges": []
            }
        
        # Try different extraction methods
        years_explicit = self._extract_explicit_years(experience_text)
        years_from_dates = self._extract_years_from_dates(experience_text)
        
        # Use the maximum value found
        if years_explicit is not None and years_from_dates is not None:
            experience_years = max(years_explicit, years_from_dates)
            method = "combined"
        elif years_explicit is not None:
            experience_years = years_explicit
            method = "explicit"
        elif years_from_dates is not None:
            experience_years = years_from_dates
            method = "date_calculation"
        else:
            experience_years = 0.0
            method = "estimation_failed"
        
        # Check if experience meets requirement
        experience_match = True
        if required_years is not None:
            experience_match = experience_years >= required_years
        
        return {
            "experience_years": round(experience_years, 1),
            "experience_match": experience_match,
            "calculation_method": method,
            "required_years": required_years,
        }
    
    def _extract_explicit_years(self, text: str) -> Optional[float]:
        """
        Extract explicit year mentions like "5 years of experience"
        
        Args:
            text: Experience text
        
        Returns:
            Maximum years mentioned or None
        """
        # Patterns for explicit year mentions
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*years?\s+(?:of\s+)?experience',
            r'experience\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:\+)?\s*years?',
            r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*yrs?\s+(?:of\s+)?experience',
            r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*years?\s+(?:in|with)',
        ]
        
        years_found = []
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                try:
                    years = float(match.group(1))
                    if 0 < years <= 50:  # Sanity check
                        years_found.append(years)
                except (ValueError, IndexError):
                    continue
        
        return max(years_found) if years_found else None
    
    def _extract_years_from_dates(self, text: str) -> Optional[float]:
        """
        Calculate total experience from date ranges
        
        Args:
            text: Experience text with date ranges
        
        Returns:
            Total years calculated from all date ranges
        """
        date_ranges = self._find_date_ranges(text)
        
        if not date_ranges:
            return None
        
        total_months = 0
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        for start_date, end_date in date_ranges:
            # Handle "Present", "Current", "Now"
            if end_date is None:
                end_year = current_year
                end_month = current_month
            else:
                end_year, end_month = end_date
            
            start_year, start_month = start_date
            
            # Calculate months between dates
            months = (end_year - start_year) * 12 + (end_month - start_month)
            
            # Sanity check
            if 0 <= months <= 600:  # Max 50 years
                total_months += months
        
        # Convert months to years
        total_years = total_months / 12.0
        
        return total_years if total_years > 0 else None
    
    def _find_date_ranges(self, text: str) -> List[Tuple[Tuple[int, int], Optional[Tuple[int, int]]]]:
        """
        Find all date ranges in text
        
        Returns:
            List of tuples: ((start_year, start_month), (end_year, end_month))
            end_date is None if "Present"/"Current"
        """
        date_ranges = []
        
        # Pattern 1: "2019 - 2023" or "2019-2023"
        pattern_year_range = r'(\d{4})\s*[-–—]\s*(\d{4}|present|current|now)'
        
        # Pattern 2: "Jan 2019 - Dec 2023" or "January 2019 - Present"
        pattern_month_year = r'(\w+)\s+(\d{4})\s*[-–—]\s*(?:(\w+)\s+)?(\d{4}|present|current|now)'
        
        text_lower = text.lower()
        
        # Extract year ranges
        for match in re.finditer(pattern_year_range, text_lower):
            start_year = int(match.group(1))
            end_str = match.group(2)
            
            if end_str in ['present', 'current', 'now']:
                end_year = None
                end_month = None
            else:
                end_year = int(end_str)
                end_month = 12  # Assume end of year
            
            # Default to January for start month if only year given
            start_month = 1
            
            if end_year is None:
                date_ranges.append(((start_year, start_month), None))
            else:
                date_ranges.append(((start_year, start_month), (end_year, end_month)))
        
        # Extract month-year ranges
        for match in re.finditer(pattern_month_year, text_lower):
            start_month_str = match.group(1)
            start_year = int(match.group(2))
            end_month_str = match.group(3)
            end_str = match.group(4)
            
            # Parse start month
            start_month = self.month_map.get(start_month_str[:3], 1)
            
            # Parse end date
            if end_str in ['present', 'current', 'now']:
                end_year = None
                end_month = None
            else:
                end_year = int(end_str)
                if end_month_str:
                    end_month = self.month_map.get(end_month_str[:3], 12)
                else:
                    end_month = 12
            
            if end_year is None:
                date_ranges.append(((start_year, start_month), None))
            else:
                date_ranges.append(((start_year, start_month), (end_year, end_month)))
        
        return date_ranges
    
    def get_experience_level(self, years: float) -> str:
        """
        Categorize experience into levels
        
        Args:
            years: Years of experience
        
        Returns:
            Experience level string
        """
        if years < 1:
            return "entry_level"
        elif years < 3:
            return "junior"
        elif years < 5:
            return "mid_level"
        elif years < 8:
            return "senior"
        elif years < 12:
            return "lead"
        else:
            return "expert"
