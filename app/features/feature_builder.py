"""
Feature Builder
Combines all extracted features into structured representation
Phase 3: Feature Engineering - MOST IMPORTANT MODULE
"""

from typing import Dict, Optional, List
from app.features.skill_extractor import SkillExtractor
from app.features.experience_extractor import ExperienceExtractor
from app.features.project_extractor import ProjectExtractor
from app.features.education_extractor import EducationExtractor
from app.utils.normalizer import Normalizer


class FeatureBuilder:
    """
    Orchestrates all feature extractors to build structured resume representation
    
    This is the core module that transforms unstructured resume text
    into machine-readable numerical and categorical features.
    
    Output is used by scoring algorithms and ML models.
    """
    
    def __init__(self):
        """Initialize all feature extractors"""
        self.skill_extractor = SkillExtractor()
        self.experience_extractor = ExperienceExtractor()
        self.project_extractor = ProjectExtractor()
        self.education_extractor = EducationExtractor()
        self.normalizer = Normalizer()
    
    def build_features(
        self,
        resume_sections: Dict[str, str],
        job_role_data: Dict
    ) -> Dict:
        """
        Build complete feature representation from resume and job role
        
        Args:
            resume_sections: Dict with sections (skills, experience, projects, education)
            job_role_data: Dict with job requirements (skills, min_experience, degree, etc.)
        
        Returns:
            Structured feature dictionary with normalized values
        """
        # Extract raw features from each section
        skills_features = self._extract_skill_features(
            resume_sections.get('skills', ''),
            job_role_data.get('required_skills', [])
        )
        
        experience_features = self._extract_experience_features(
            resume_sections.get('experience', ''),
            job_role_data.get('min_experience_years', None)
        )
        
        project_features = self._extract_project_features(
            resume_sections.get('projects', ''),
            job_role_data.get('required_skills', [])
        )
        
        education_features = self._extract_education_features(
            resume_sections.get('education', ''),
            job_role_data.get('required_degree', None)
        )
        
        # Combine all features
        combined_features = {
            # Skill features
            'skill_match_percent': skills_features['skill_match_percent'],
            'matched_skills': skills_features['matched_skills'],
            'missing_skills': skills_features['missing_skills'],
            'matched_skill_count': skills_features['matched_count'],
            'required_skill_count': skills_features['required_count'],
            
            # Experience features
            'experience_years': experience_features['experience_years'],
            'experience_match': experience_features['experience_match'],
            'experience_level': experience_features.get('experience_level', 'unknown'),
            
            # Project features
            'project_count': project_features['project_count'],
            'relevant_projects': project_features['relevant_projects'],
            'project_score': project_features['project_score'],
            
            # Education features
            'highest_degree': education_features['highest_degree'],
            'education_match': education_features['education_match'],
            'degree_level': education_features['degree_level'],
            'field_of_study': education_features['field_of_study'],
            
            # Normalized features (for ML models)
            'normalized_features': self._normalize_all_features(
                skills_features,
                experience_features,
                project_features,
                education_features,
                job_role_data
            ),
        }
        
        # Add metadata
        combined_features['metadata'] = {
            'job_role': job_role_data.get('role_name', 'Unknown'),
            'extraction_method': 'rule_based_feature_engineering',
            'feature_count': len(combined_features['normalized_features']),
        }
        
        return combined_features
    
    def _extract_skill_features(
        self,
        skills_text: str,
        required_skills: List[str]
    ) -> Dict:
        """Extract skill-related features"""
        return self.skill_extractor.extract_skills(skills_text, required_skills)
    
    def _extract_experience_features(
        self,
        experience_text: str,
        required_years: Optional[float]
    ) -> Dict:
        """Extract experience-related features"""
        features = self.experience_extractor.extract_experience(
            experience_text,
            required_years
        )
        
        # Add experience level categorization
        features['experience_level'] = self.experience_extractor.get_experience_level(
            features['experience_years']
        )
        
        return features
    
    def _extract_project_features(
        self,
        projects_text: str,
        job_keywords: List[str]
    ) -> Dict:
        """Extract project-related features"""
        features = self.project_extractor.extract_projects(
            projects_text,
            job_keywords
        )
        
        # Add project domains
        features['project_domains'] = self.project_extractor.identify_project_domains(
            projects_text
        )
        
        return features
    
    def _extract_education_features(
        self,
        education_text: str,
        required_degree: Optional[str]
    ) -> Dict:
        """Extract education-related features"""
        features = self.education_extractor.extract_education(
            education_text,
            required_degree
        )
        
        # Add STEM background check
        features['has_stem_background'] = self.education_extractor.check_stem_background(
            features['field_of_study']
        )
        
        # Add university names
        features['universities'] = self.education_extractor.extract_university_names(
            education_text
        )
        
        return features
    
    def _normalize_all_features(
        self,
        skills_features: Dict,
        experience_features: Dict,
        project_features: Dict,
        education_features: Dict,
        job_role_data: Dict
    ) -> Dict:
        """
        Normalize all features to 0-1 range for ML models
        
        Returns:
            Dict with all normalized feature values
        """
        normalized = {}
        
        # Skill match (already 0-1)
        normalized['skill_match'] = skills_features['skill_match_percent']
        
        # Experience (normalize based on job requirements)
        normalized['experience'] = self.normalizer.normalize_experience(
            experience_features['experience_years'],
            job_role_data.get('min_experience_years', None)
        )
        
        # Projects (already 0-1)
        normalized['project_score'] = project_features['project_score']
        
        # Project count (diminishing returns after 3)
        normalized['project_count_normalized'] = self.normalizer.normalize_project_count(
            project_features['project_count']
        )
        
        # Education (normalize degree level)
        required_degree = job_role_data.get('required_degree', 'bachelor')
        required_level = {
            'high_school': 1,
            'diploma': 2,
            'associate': 3,
            'bachelor': 4,
            'master': 5,
            'phd': 6
        }.get(required_degree.lower() if required_degree else 'bachelor', 4)
        
        normalized['education'] = self.normalizer.normalize_education(
            education_features['degree_level'],
            required_level
        )
        
        # Boolean features (convert to 0/1)
        normalized['experience_met'] = 1.0 if experience_features['experience_match'] else 0.0
        normalized['education_met'] = 1.0 if education_features['education_match'] else 0.0
        normalized['has_projects'] = 1.0 if project_features['has_projects'] else 0.0
        normalized['has_stem_background'] = 1.0 if education_features['has_stem_background'] else 0.0
        
        # Derived features
        normalized['skill_coverage'] = self.normalizer.clamp(
            skills_features['matched_count'] / max(skills_features['required_count'], 1)
        )
        
        normalized['project_relevance'] = self.normalizer.clamp(
            project_features['relevant_projects'] / max(project_features['project_count'], 1)
            if project_features['project_count'] > 0 else 0.0
        )
        
        return normalized
    
    def get_feature_summary(self, features: Dict) -> str:
        """
        Generate human-readable summary of extracted features
        
        Args:
            features: Feature dictionary from build_features()
        
        Returns:
            Text summary
        """
        summary_parts = []
        
        # Skills
        summary_parts.append(
            f"Skills: {features['matched_skill_count']}/{features['required_skill_count']} matched "
            f"({features['skill_match_percent']*100:.0f}%)"
        )
        
        # Experience
        summary_parts.append(
            f"Experience: {features['experience_years']} years "
            f"({'✓' if features['experience_match'] else '✗'})"
        )
        
        # Projects
        summary_parts.append(
            f"Projects: {features['project_count']} total, "
            f"{features['relevant_projects']} relevant "
            f"(score: {features['project_score']})"
        )
        
        # Education
        degree_label = self.education_extractor.get_degree_label(
            features['highest_degree']
        ) if features['highest_degree'] else 'Not specified'
        summary_parts.append(
            f"Education: {degree_label} "
            f"({'✓' if features['education_match'] else '✗'})"
        )
        
        return " | ".join(summary_parts)
    
    def get_feature_vector(self, features: Dict) -> List[float]:
        """
        Extract feature vector for ML models
        
        Args:
            features: Feature dictionary from build_features()
        
        Returns:
            List of normalized feature values
        """
        normalized = features['normalized_features']
        
        # Define feature order (important for ML models)
        feature_order = [
            'skill_match',
            'experience',
            'project_score',
            'education',
            'experience_met',
            'education_met',
            'has_projects',
            'skill_coverage',
            'project_relevance',
        ]
        
        return [normalized.get(key, 0.0) for key in feature_order]
