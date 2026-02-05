"""
Section Extractor Service
Identifies and extracts resume sections using regex and heuristics
Phase 2: Section Segmentation Layer
"""

import re
from typing import Dict, List, Tuple


class SectionExtractorService:
    """
    Service for extracting structured sections from resume text
    Identifies: Skills, Experience, Projects, Education, Certifications, Summary
    """
    
    def __init__(self):
        """Initialize section patterns and keywords"""
        
        # Section header patterns (case-insensitive)
        self.section_patterns = {
            'skills': [
                r'\b(technical\s+)?skills?\b',
                r'\bcore\s+competencies\b',
                r'\bexpertise\b',
                r'\btechnologies\b',
                r'\bproficiencies\b',
            ],
            'experience': [
                r'\b(professional\s+)?(work\s+)?experience\b',
                r'\bemployment\s+history\b',
                r'\bwork\s+history\b',
                r'\bcareer\s+summary\b',
            ],
            'projects': [
                r'\bprojects?\b',
                r'\bkey\s+projects?\b',
                r'\bacademic\s+projects?\b',
                r'\bpersonal\s+projects?\b',
            ],
            'education': [
                r'\beducation(al)?\s+(background)?\b',
                r'\bacademic\s+(qualifications?)?\b',
                r'\bdegree\b',
            ],
            'certifications': [
                r'\bcertifications?\b',
                r'\blicenses?\b',
                r'\bprofessional\s+certifications?\b',
                r'\bcredentials?\b',
            ],
            'summary': [
                r'\b(professional\s+)?summary\b',
                r'\bprofile\b',
                r'\bobjective\b',
                r'\babout\s+me\b',
                r'\bcareer\s+objective\b',
            ],
        }
        
        # Compile regex patterns
        self.compiled_patterns = {}
        for section, patterns in self.section_patterns.items():
            self.compiled_patterns[section] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract all identifiable sections from resume text
        
        Args:
            text: Raw resume text
        
        Returns:
            Dictionary mapping section names to their content
        """
        # Split text into lines for processing
        lines = text.split('\n')
        
        # Find section boundaries
        section_boundaries = self._find_section_boundaries(lines)
        
        # Extract content for each section
        sections = {}
        for section_name, (start_idx, end_idx) in section_boundaries.items():
            section_content = '\n'.join(lines[start_idx:end_idx]).strip()
            if section_content:
                sections[section_name] = section_content
        
        return sections
    
    def _find_section_boundaries(self, lines: List[str]) -> Dict[str, Tuple[int, int]]:
        """
        Identify line ranges for each section
        
        Args:
            lines: List of text lines
        
        Returns:
            Dict mapping section names to (start_line, end_line) tuples
        """
        boundaries = {}
        section_starts = []  # List of (line_idx, section_name)
        
        # Scan for section headers
        for idx, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if line matches any section pattern
            for section_name, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(line_stripped):
                        # Potential section header found
                        # Verify it's likely a header (short line, possibly with formatting)
                        if len(line_stripped) < 50 and not self._looks_like_content(line_stripped):
                            section_starts.append((idx, section_name))
                            break
                if section_starts and section_starts[-1][0] == idx:
                    break  # Found match, move to next line
        
        # Sort section starts by line index
        section_starts.sort(key=lambda x: x[0])
        
        # Define section boundaries
        for i, (start_idx, section_name) in enumerate(section_starts):
            # Section content starts on next line
            content_start = start_idx + 1
            
            # Section ends at next section or end of document
            if i + 1 < len(section_starts):
                content_end = section_starts[i + 1][0]
            else:
                content_end = len(lines)
            
            # Store boundaries (skip empty sections)
            if content_end > content_start:
                boundaries[section_name] = (content_start, content_end)
        
        return boundaries
    
    def _looks_like_content(self, line: str) -> bool:
        """
        Heuristic to determine if line is content rather than a header
        
        Args:
            line: Text line to check
        
        Returns:
            True if line appears to be content, False if likely a header
        """
        # Content indicators: contains multiple sentences, bullets, dates
        content_indicators = [
            len(line.split()) > 10,  # Long line
            line.count('.') >= 2,  # Multiple sentences
            re.search(r'\d{4}', line),  # Contains year
            line.startswith('•') or line.startswith('-'),  # Bullet point
            re.search(r'https?://', line),  # Contains URL
        ]
        
        return any(content_indicators)
    
    def get_section(self, text: str, section_name: str) -> str:
        """
        Extract a specific section by name
        
        Args:
            text: Raw resume text
            section_name: Name of section to extract
        
        Returns:
            Section content or empty string if not found
        """
        sections = self.extract_sections(text)
        return sections.get(section_name, '')
    
    def list_found_sections(self, text: str) -> List[str]:
        """
        Get list of section names found in resume
        
        Args:
            text: Raw resume text
        
        Returns:
            List of section names detected
        """
        sections = self.extract_sections(text)
        return list(sections.keys())
