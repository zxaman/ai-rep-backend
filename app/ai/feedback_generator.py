"""
Feedback Generator for Phase 6

This module combines structured analysis data with AI-generated feedback
to produce comprehensive, actionable resume improvement suggestions.

KEY PRINCIPLES:
1. Structured data comes from Phases 2-5 (parsing, features, similarity, scoring)
2. AI is used ONLY for explanations and suggestions
3. AI does NOT calculate scores or make decisions
4. Output is JSON-formatted for easy frontend consumption
"""

import logging
from typing import Dict, List, Any, Optional
from app.ai.prompt_templates import PromptTemplateManager
from app.ai.ai_client import AIClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackGenerator:
    """
    Generates comprehensive feedback by combining structured analysis
    with AI-powered explanations and suggestions.
    
    This class orchestrates:
    1. Prompt generation (from templates)
    2. AI response generation (via AIClient)
    3. Response parsing and structuring
    4. Final JSON formatting
    """
    
    def __init__(
        self,
        ai_client: Optional[AIClient] = None,
        prompt_manager: Optional[PromptTemplateManager] = None
    ):
        """
        Initialize feedback generator.
        
        Args:
            ai_client: AIClient instance (if None, creates default)
            prompt_manager: PromptTemplateManager instance (if None, creates default)
        """
        self.ai_client = ai_client or AIClient()
        self.prompt_manager = prompt_manager or PromptTemplateManager()
        
        logger.info("Feedback generator initialized")
    
    def generate_comprehensive_feedback(
        self,
        job_role: str,
        final_score: int,
        interpretation: str,
        score_breakdown: Dict[str, float],
        matched_skills: List[str],
        missing_skills: List[str],
        experience_years: float,
        required_experience: float,
        experience_match: bool,
        project_count: int,
        relevant_projects: int
    ) -> Dict[str, Any]:
        """
        Generate comprehensive feedback with all components.
        
        Args:
            job_role: Target job role
            final_score: Final resume score (0-100)
            interpretation: Score interpretation
            score_breakdown: Component scores dict
            matched_skills: List of matched skills
            missing_skills: List of missing skills
            experience_years: Candidate's experience
            required_experience: Required experience
            experience_match: Experience requirement met
            project_count: Total projects
            relevant_projects: Relevant projects
        
        Returns:
            Structured feedback dict with AI-generated suggestions
        """
        try:
            # Generate prompt
            prompt = self.prompt_manager.generate_comprehensive_prompt(
                job_role=job_role,
                final_score=final_score,
                interpretation=interpretation,
                score_breakdown=score_breakdown,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                experience_years=experience_years,
                required_experience=required_experience,
                experience_match=experience_match,
                project_count=project_count,
                relevant_projects=relevant_projects
            )
            
            # Get system prompt
            system_prompt = self.prompt_manager.get_system_prompt()
            
            # Generate AI response
            logger.info(f"Generating comprehensive feedback for {job_role}")
            ai_response = self.ai_client.generate_response(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.7,
                max_tokens=1500
            )
            
            # Structure the response
            feedback = {
                'feedback_type': 'comprehensive',
                'job_role': job_role,
                'score_context': {
                    'final_score': final_score,
                    'interpretation': interpretation,
                    'breakdown': score_breakdown
                },
                'ai_feedback': {
                    'raw_response': ai_response,
                    'summary': self._extract_section(ai_response, 'RESUME SUMMARY'),
                    'skill_gap_analysis': self._extract_section(ai_response, 'SKILL GAP ANALYSIS'),
                    'improvement_suggestions': self._extract_bullet_points(ai_response, 'IMPROVEMENT SUGGESTIONS'),
                    'ats_tips': self._extract_bullet_points(ai_response, 'ATS OPTIMIZATION TIPS')
                },
                'structured_data': {
                    'matched_skills': matched_skills,
                    'missing_skills': missing_skills,
                    'experience': {
                        'candidate_years': experience_years,
                        'required_years': required_experience,
                        'meets_requirement': experience_match
                    },
                    'projects': {
                        'total': project_count,
                        'relevant': relevant_projects
                    }
                }
            }
            
            logger.info("Comprehensive feedback generated successfully")
            return feedback
        
        except Exception as e:
            logger.error(f"Error generating comprehensive feedback: {str(e)}")
            return self._generate_fallback_feedback(
                job_role=job_role,
                final_score=final_score,
                interpretation=interpretation,
                error=str(e)
            )
    
    def generate_skill_feedback(
        self,
        job_role: str,
        matched_skills: List[str],
        missing_skills: List[str],
        skill_match_percent: float
    ) -> Dict[str, Any]:
        """
        Generate skill-focused feedback.
        
        Args:
            job_role: Target job role
            matched_skills: List of matched skills
            missing_skills: List of missing skills
            skill_match_percent: Skill match percentage
        
        Returns:
            Skill-focused feedback dict
        """
        try:
            prompt = self.prompt_manager.generate_skill_feedback_prompt(
                job_role=job_role,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                skill_match_percent=skill_match_percent
            )
            
            system_prompt = self.prompt_manager.get_system_prompt()
            
            logger.info(f"Generating skill feedback for {job_role}")
            ai_response = self.ai_client.generate_response(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.7,
                max_tokens=1000
            )
            
            return {
                'feedback_type': 'skill_focused',
                'job_role': job_role,
                'skill_match_percent': skill_match_percent,
                'ai_feedback': ai_response,
                'matched_skills': matched_skills,
                'missing_skills': missing_skills
            }
        
        except Exception as e:
            logger.error(f"Error generating skill feedback: {str(e)}")
            return {'error': str(e), 'feedback_type': 'skill_focused'}
    
    def generate_concise_feedback(
        self,
        job_role: str,
        final_score: int,
        interpretation: str,
        missing_skills: List[str]
    ) -> Dict[str, Any]:
        """
        Generate concise feedback for quick responses.
        
        Args:
            job_role: Target job role
            final_score: Final resume score
            interpretation: Score interpretation
            missing_skills: List of missing skills
        
        Returns:
            Concise feedback dict
        """
        try:
            prompt = self.prompt_manager.generate_concise_feedback_prompt(
                job_role=job_role,
                final_score=final_score,
                interpretation=interpretation,
                missing_skills=missing_skills
            )
            
            system_prompt = self.prompt_manager.get_system_prompt()
            
            logger.info(f"Generating concise feedback for {job_role}")
            ai_response = self.ai_client.generate_response(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.5,  # Lower temperature for more focused response
                max_tokens=500
            )
            
            return {
                'feedback_type': 'concise',
                'job_role': job_role,
                'final_score': final_score,
                'interpretation': interpretation,
                'ai_feedback': ai_response,
                'missing_skills': missing_skills[:5]  # Top 5 only
            }
        
        except Exception as e:
            logger.error(f"Error generating concise feedback: {str(e)}")
            return {'error': str(e), 'feedback_type': 'concise'}
    
    def generate_ats_optimization_feedback(
        self,
        job_role: str,
        final_score: int,
        matched_skills: List[str],
        missing_skills: List[str]
    ) -> Dict[str, Any]:
        """
        Generate ATS optimization feedback.
        
        Args:
            job_role: Target job role
            final_score: Final resume score
            matched_skills: Matched skills
            missing_skills: Missing skills
        
        Returns:
            ATS optimization feedback dict
        """
        try:
            prompt = self.prompt_manager.generate_ats_optimization_prompt(
                job_role=job_role,
                final_score=final_score,
                matched_skills=matched_skills,
                missing_skills=missing_skills
            )
            
            system_prompt = self.prompt_manager.get_system_prompt()
            
            logger.info(f"Generating ATS optimization feedback for {job_role}")
            ai_response = self.ai_client.generate_response(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.6,
                max_tokens=800
            )
            
            return {
                'feedback_type': 'ats_optimization',
                'job_role': job_role,
                'current_score': final_score,
                'ai_feedback': ai_response,
                'matched_skills': matched_skills,
                'missing_skills': missing_skills
            }
        
        except Exception as e:
            logger.error(f"Error generating ATS feedback: {str(e)}")
            return {'error': str(e), 'feedback_type': 'ats_optimization'}
    
    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """
        Extract a specific section from AI response.
        
        Args:
            text: Full AI response text
            section_name: Name of section to extract
        
        Returns:
            Extracted section text or None
        """
        try:
            # Look for section header
            start_marker = f"{section_name}:"
            if start_marker not in text:
                start_marker = f"{section_name.upper()}:"
            
            if start_marker in text:
                start_idx = text.index(start_marker) + len(start_marker)
                
                # Find next section or end of text
                remaining = text[start_idx:]
                
                # Look for next numbered section
                next_section_idx = len(remaining)
                for i in range(1, 10):
                    marker = f"\n{i}."
                    if marker in remaining:
                        next_section_idx = min(next_section_idx, remaining.index(marker))
                
                return remaining[:next_section_idx].strip()
            
            return None
        
        except Exception as e:
            logger.warning(f"Error extracting section {section_name}: {str(e)}")
            return None
    
    def _extract_bullet_points(self, text: str, section_name: str) -> List[str]:
        """
        Extract bullet points from a section.
        
        Args:
            text: Full AI response text
            section_name: Name of section containing bullet points
        
        Returns:
            List of bullet point strings
        """
        section_text = self._extract_section(text, section_name)
        if not section_text:
            return []
        
        # Extract lines starting with - or *
        bullet_points = []
        for line in section_text.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('*') or line.startswith('•'):
                # Remove bullet marker
                point = line[1:].strip()
                if point:
                    bullet_points.append(point)
        
        return bullet_points
    
    def _generate_fallback_feedback(
        self,
        job_role: str,
        final_score: int,
        interpretation: str,
        error: str
    ) -> Dict[str, Any]:
        """
        Generate fallback feedback when AI generation fails.
        
        Args:
            job_role: Target job role
            final_score: Final score
            interpretation: Score interpretation
            error: Error message
        
        Returns:
            Fallback feedback dict
        """
        logger.warning("Using fallback feedback due to AI error")
        
        # Generate generic feedback based on score
        if final_score >= 85:
            summary = f"Excellent match for {job_role} position. Strong qualifications across all areas."
            suggestions = [
                "Continue highlighting your key achievements",
                "Ensure all projects have quantifiable results",
                "Keep resume updated with latest skills"
            ]
        elif final_score >= 70:
            summary = f"Good match for {job_role} position with solid qualifications."
            suggestions = [
                "Consider adding more relevant projects",
                "Highlight specific achievements with metrics",
                "Address any missing critical skills through learning"
            ]
        elif final_score >= 55:
            summary = f"Moderate match for {job_role} position with room for improvement."
            suggestions = [
                "Focus on acquiring missing critical skills",
                "Add more relevant project experience",
                "Strengthen technical skills section",
                "Consider certifications in key areas"
            ]
        else:
            summary = f"Limited match for {job_role} position. Significant skill gaps present."
            suggestions = [
                "Prioritize learning missing critical skills",
                "Build portfolio projects in relevant technologies",
                "Consider internships or entry-level positions first",
                "Take online courses to fill knowledge gaps"
            ]
        
        return {
            'feedback_type': 'fallback',
            'job_role': job_role,
            'score_context': {
                'final_score': final_score,
                'interpretation': interpretation
            },
            'ai_feedback': {
                'summary': summary,
                'improvement_suggestions': suggestions,
                'note': 'AI-generated feedback unavailable. Showing rule-based suggestions.',
                'error': error
            }
        }
    
    def format_for_api_response(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format feedback for API response.
        
        Args:
            feedback: Raw feedback dict
        
        Returns:
            Formatted feedback dict ready for JSON serialization
        """
        return {
            'success': 'error' not in feedback,
            'feedback_type': feedback.get('feedback_type', 'unknown'),
            'data': feedback
        }


# Example usage
if __name__ == "__main__":
    # Example: Generate comprehensive feedback
    try:
        generator = FeedbackGenerator()
        
        feedback = generator.generate_comprehensive_feedback(
            job_role="Data Scientist",
            final_score=75,
            interpretation="Good",
            score_breakdown={
                'skill_match': 30.0,
                'semantic_similarity': 18.75,
                'experience': 15.0,
                'project_score': 11.25
            },
            matched_skills=["Python", "Machine Learning", "SQL"],
            missing_skills=["Deep Learning", "TensorFlow", "AWS"],
            experience_years=3.5,
            required_experience=3.0,
            experience_match=True,
            project_count=5,
            relevant_projects=3
        )
        
        print("Feedback generated successfully!")
        print(f"Type: {feedback['feedback_type']}")
        
    except Exception as e:
        print(f"Error: {e}")
