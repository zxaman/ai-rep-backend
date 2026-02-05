"""
Skill Extractor
Extracts and matches skills from resume against job role requirements
Phase 3: Feature Engineering
"""

import re
from typing import Dict, List, Set


class SkillExtractor:
    """
    Rule-based skill extraction and matching service
    
    Matches resume skills against job role skill requirements using:
    - Case-insensitive keyword matching
    - Basic synonym mapping
    - Skill match percentage calculation
    """
    
    def __init__(self):
        """Initialize skill extractor with synonym mappings"""
        
        # Common skill synonyms and variations
        self.skill_synonyms = {
            # Programming Languages
            'javascript': ['js', 'javascript', 'ecmascript'],
            'typescript': ['ts', 'typescript'],
            'python': ['python', 'python3', 'py'],
            'java': ['java', 'java8', 'java11', 'java17'],
            'csharp': ['c#', 'csharp', 'c sharp', '.net'],
            'cpp': ['c++', 'cpp', 'cplusplus'],
            
            # Frameworks
            'react': ['react', 'reactjs', 'react.js'],
            'angular': ['angular', 'angularjs', 'angular2+'],
            'vue': ['vue', 'vuejs', 'vue.js'],
            'nodejs': ['node', 'nodejs', 'node.js'],
            'django': ['django', 'django rest framework', 'drf'],
            'flask': ['flask', 'flask-restful'],
            'springboot': ['spring boot', 'springboot', 'spring'],
            
            # Databases
            'sql': ['sql', 'mysql', 'postgresql', 'postgres', 'mssql'],
            'nosql': ['nosql', 'mongodb', 'cassandra', 'dynamodb'],
            'mongodb': ['mongodb', 'mongo'],
            
            # Cloud
            'aws': ['aws', 'amazon web services'],
            'azure': ['azure', 'microsoft azure'],
            'gcp': ['gcp', 'google cloud', 'google cloud platform'],
            
            # ML/AI
            'machinelearning': ['machine learning', 'ml', 'ai'],
            'deeplearning': ['deep learning', 'dl', 'neural networks'],
            'tensorflow': ['tensorflow', 'tf'],
            'pytorch': ['pytorch', 'torch'],
            'sklearn': ['scikit-learn', 'sklearn', 'scikit learn'],
            
            # DevOps
            'docker': ['docker', 'containerization'],
            'kubernetes': ['kubernetes', 'k8s'],
            'jenkins': ['jenkins', 'ci/cd'],
            'git': ['git', 'github', 'gitlab', 'bitbucket'],
            
            # Data Tools
            'tableau': ['tableau', 'data visualization'],
            'powerbi': ['power bi', 'powerbi', 'pbi'],
            'excel': ['excel', 'microsoft excel', 'spreadsheet'],
            
            # Testing
            'testing': ['testing', 'unit testing', 'integration testing'],
            'selenium': ['selenium', 'automated testing'],
            'jest': ['jest', 'testing framework'],
        }
        
        # Reverse mapping for lookup
        self.normalized_skills = {}
        for normalized, variations in self.skill_synonyms.items():
            for variation in variations:
                self.normalized_skills[variation.lower()] = normalized
    
    def extract_skills(
        self,
        resume_skills_text: str,
        job_role_skills: List[str]
    ) -> Dict:
        """
        Extract and match skills from resume against job role requirements
        
        Args:
            resume_skills_text: Skills section from resume
            job_role_skills: List of required skills for job role
        
        Returns:
            Dict with matched_skills, missing_skills, skill_match_percent
        """
        # Normalize and extract resume skills
        resume_skill_set = self._extract_skill_keywords(resume_skills_text)
        
        # Normalize job role skills
        job_skill_set = self._normalize_skill_list(job_role_skills)
        
        # Find matches
        matched_skills = []
        missing_skills = []
        
        for job_skill in job_role_skills:
            normalized_job_skill = self._normalize_skill(job_skill)
            
            if normalized_job_skill in resume_skill_set:
                matched_skills.append(job_skill)
            else:
                missing_skills.append(job_skill)
        
        # Calculate match percentage
        total_required = len(job_role_skills)
        if total_required == 0:
            skill_match_percent = 0.0
        else:
            skill_match_percent = len(matched_skills) / total_required
        
        return {
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "skill_match_percent": round(skill_match_percent, 2),
            "matched_count": len(matched_skills),
            "required_count": total_required,
        }
    
    def _extract_skill_keywords(self, text: str) -> Set[str]:
        """
        Extract skill keywords from resume text
        
        Args:
            text: Skills section text
        
        Returns:
            Set of normalized skill keywords
        """
        if not text:
            return set()
        
        text = text.lower()
        skills_found = set()
        
        # Split by common separators
        # Handle: Python, Java | React & Node.js / Docker
        separators = r'[,|\n/•·\-]'
        tokens = re.split(separators, text)
        
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            
            # Remove common prefixes like "Experience with", "Knowledge of"
            token = re.sub(r'^(experience\s+(with|in)|knowledge\s+of|proficient\s+in|skilled\s+in)\s+', '', token)
            
            # Check for multi-word skills first (e.g., "machine learning", "node.js")
            normalized = self._normalize_skill(token)
            if normalized:
                skills_found.add(normalized)
            
            # Also check individual words (for cases like "Python and Java")
            words = re.findall(r'\b[a-z0-9#+.]+\b', token)
            for word in words:
                if len(word) > 1:  # Skip single letters
                    normalized_word = self._normalize_skill(word)
                    if normalized_word:
                        skills_found.add(normalized_word)
        
        return skills_found
    
    def _normalize_skill(self, skill: str) -> str:
        """
        Normalize skill to standard form using synonym mapping
        
        Args:
            skill: Raw skill string
        
        Returns:
            Normalized skill name or original if no mapping found
        """
        skill_lower = skill.lower().strip()
        
        # Check if skill exists in synonym mapping
        if skill_lower in self.normalized_skills:
            return self.normalized_skills[skill_lower]
        
        # Check if any synonym variation is contained in the skill
        for variation, normalized in self.normalized_skills.items():
            if variation in skill_lower or skill_lower in variation:
                return normalized
        
        # Return cleaned original if no mapping found
        return re.sub(r'[^\w\s#+.]', '', skill_lower).strip()
    
    def _normalize_skill_list(self, skills: List[str]) -> Set[str]:
        """
        Normalize a list of skills
        
        Args:
            skills: List of skill strings
        
        Returns:
            Set of normalized skills
        """
        normalized = set()
        for skill in skills:
            norm_skill = self._normalize_skill(skill)
            if norm_skill:
                normalized.add(norm_skill)
        return normalized
    
    def get_skill_categories(self, matched_skills: List[str]) -> Dict[str, List[str]]:
        """
        Categorize matched skills into groups
        
        Args:
            matched_skills: List of matched skill names
        
        Returns:
            Dict mapping category names to skill lists
        """
        categories = {
            'programming': [],
            'frameworks': [],
            'databases': [],
            'cloud': [],
            'tools': [],
            'other': []
        }
        
        programming_keywords = ['python', 'java', 'javascript', 'typescript', 'cpp', 'csharp', 'go', 'ruby']
        framework_keywords = ['react', 'angular', 'vue', 'django', 'flask', 'springboot', 'nodejs']
        database_keywords = ['sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis']
        cloud_keywords = ['aws', 'azure', 'gcp', 'cloud']
        tool_keywords = ['docker', 'kubernetes', 'git', 'jenkins', 'tableau', 'powerbi']
        
        for skill in matched_skills:
            skill_norm = self._normalize_skill(skill)
            
            if skill_norm in programming_keywords:
                categories['programming'].append(skill)
            elif skill_norm in framework_keywords:
                categories['frameworks'].append(skill)
            elif skill_norm in database_keywords:
                categories['databases'].append(skill)
            elif skill_norm in cloud_keywords:
                categories['cloud'].append(skill)
            elif skill_norm in tool_keywords:
                categories['tools'].append(skill)
            else:
                categories['other'].append(skill)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
