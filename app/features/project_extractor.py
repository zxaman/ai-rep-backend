"""
Project Extractor
Evaluates project count and relevance to job role
Phase 3: Feature Engineering
"""

import re
from typing import Dict, List, Set


class ProjectExtractor:
    """
    Rule-based project evaluation service
    
    Evaluates projects based on:
    - Project count
    - Keyword matching against job role requirements
    - Project complexity indicators
    """
    
    def __init__(self):
        """Initialize project extractor"""
        
        # Common project-related keywords that indicate real projects
        self.project_indicators = [
            'built', 'developed', 'created', 'designed', 'implemented',
            'deployed', 'architected', 'engineered', 'launched',
            'led', 'managed', 'contributed', 'collaborated'
        ]
        
        # Technology domain keywords for relevance matching
        self.domain_keywords = {
            'web': ['web', 'website', 'frontend', 'backend', 'fullstack', 'api', 'rest', 'graphql'],
            'mobile': ['mobile', 'android', 'ios', 'react native', 'flutter', 'app'],
            'ml': ['machine learning', 'deep learning', 'neural network', 'ai', 'model', 'prediction'],
            'data': ['data', 'analytics', 'visualization', 'dashboard', 'etl', 'pipeline', 'warehouse'],
            'cloud': ['cloud', 'aws', 'azure', 'gcp', 'kubernetes', 'docker', 'microservices'],
            'devops': ['devops', 'ci/cd', 'jenkins', 'deployment', 'automation', 'infrastructure'],
        }
    
    def extract_projects(
        self,
        projects_text: str,
        job_role_keywords: List[str] = None
    ) -> Dict:
        """
        Extract and evaluate projects from resume
        
        Args:
            projects_text: Projects section from resume
            job_role_keywords: Keywords relevant to the job role
        
        Returns:
            Dict with project_count, relevant_projects, project_score
        """
        if not projects_text:
            return {
                "project_count": 0,
                "relevant_projects": 0,
                "project_score": 0.0,
                "has_projects": False
            }
        
        # Count total projects
        project_count = self._count_projects(projects_text)
        
        # Evaluate project relevance
        relevant_count = 0
        if job_role_keywords:
            relevant_count = self._count_relevant_projects(
                projects_text,
                job_role_keywords
            )
        
        # Calculate project score (0-1 range)
        # Score based on: having projects (0.3) + quantity (0.3) + relevance (0.4)
        has_projects_score = 0.3 if project_count > 0 else 0.0
        
        # Quantity score: 0-0.3 based on project count (diminishing returns)
        if project_count == 0:
            quantity_score = 0.0
        elif project_count == 1:
            quantity_score = 0.15
        elif project_count == 2:
            quantity_score = 0.25
        else:  # 3+
            quantity_score = 0.3
        
        # Relevance score: 0-0.4 based on relevant project ratio
        if project_count > 0:
            relevance_ratio = relevant_count / project_count
            relevance_score = 0.4 * relevance_ratio
        else:
            relevance_score = 0.0
        
        project_score = has_projects_score + quantity_score + relevance_score
        
        return {
            "project_count": project_count,
            "relevant_projects": relevant_count,
            "project_score": round(project_score, 2),
            "has_projects": project_count > 0,
        }
    
    def _count_projects(self, text: str) -> int:
        """
        Count number of projects in text
        
        Args:
            text: Projects section text
        
        Returns:
            Estimated number of projects
        """
        if not text:
            return 0
        
        text_lower = text.lower()
        
        # Method 1: Count bullet points / numbered lists
        bullet_patterns = [
            r'^\s*[\-•●○▪▫]',  # Bullet points
            r'^\s*\d+\.',       # Numbered lists
        ]
        
        bullet_count = 0
        for line in text.split('\n'):
            for pattern in bullet_patterns:
                if re.match(pattern, line):
                    bullet_count += 1
                    break
        
        # Method 2: Count project indicator keywords (built, developed, etc.)
        # Each strong indicator likely represents a different project
        indicator_count = 0
        for indicator in self.project_indicators[:5]:  # Use top 5 indicators
            # Count occurrences of strong action verbs
            pattern = r'\b' + indicator + r'\b'
            matches = len(re.findall(pattern, text_lower))
            indicator_count += matches
        
        # Method 3: Count section headings that look like project titles
        # Look for capitalized lines that aren't too long
        title_count = 0
        for line in text.split('\n'):
            line_stripped = line.strip()
            if line_stripped and len(line_stripped) < 100:
                # Check if line starts with capital or is all caps
                if line_stripped[0].isupper() or line_stripped.isupper():
                    # Likely a project title
                    if not any(indicator in line_stripped.lower() for indicator in self.project_indicators):
                        title_count += 1
        
        # Use the maximum estimate, but cap at reasonable number
        estimated_count = max(bullet_count, min(indicator_count, 10), title_count)
        
        # If we found very few projects but there's substantial text, estimate at least 1
        if estimated_count == 0 and len(text.strip()) > 100:
            estimated_count = 1
        
        return min(estimated_count, 15)  # Cap at 15 projects
    
    def _count_relevant_projects(
        self,
        text: str,
        job_keywords: List[str]
    ) -> int:
        """
        Count projects that mention job-relevant keywords
        
        Args:
            text: Projects section text
            job_keywords: Keywords related to job role
        
        Returns:
            Count of projects mentioning relevant keywords
        """
        if not text or not job_keywords:
            return 0
        
        text_lower = text.lower()
        
        # Split text into project segments (by bullets/numbers or paragraphs)
        project_segments = self._split_into_projects(text)
        
        # Normalize job keywords
        normalized_keywords = set()
        for keyword in job_keywords:
            normalized_keywords.add(keyword.lower().strip())
        
        # Count how many segments contain relevant keywords
        relevant_count = 0
        for segment in project_segments:
            segment_lower = segment.lower()
            
            # Check if segment contains any job keyword
            for keyword in normalized_keywords:
                if keyword in segment_lower:
                    relevant_count += 1
                    break  # Count each segment only once
        
        return relevant_count
    
    def _split_into_projects(self, text: str) -> List[str]:
        """
        Split projects text into individual project segments
        
        Args:
            text: Projects section text
        
        Returns:
            List of project text segments
        """
        segments = []
        current_segment = []
        
        for line in text.split('\n'):
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # Check if this is a new project (bullet, number, or title)
            is_new_project = (
                re.match(r'^\s*[\-•●○▪▫]', line) or
                re.match(r'^\s*\d+\.', line) or
                (line_stripped[0].isupper() and len(line_stripped) < 100)
            )
            
            if is_new_project and current_segment:
                # Save previous segment and start new one
                segments.append('\n'.join(current_segment))
                current_segment = [line_stripped]
            else:
                current_segment.append(line_stripped)
        
        # Add final segment
        if current_segment:
            segments.append('\n'.join(current_segment))
        
        return segments
    
    def identify_project_domains(self, text: str) -> List[str]:
        """
        Identify technical domains of projects
        
        Args:
            text: Projects section text
        
        Returns:
            List of identified domains
        """
        if not text:
            return []
        
        text_lower = text.lower()
        identified_domains = []
        
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    identified_domains.append(domain)
                    break  # Only add domain once
        
        return identified_domains
