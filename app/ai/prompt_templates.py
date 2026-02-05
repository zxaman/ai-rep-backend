"""
Prompt Template Manager for Phase 6

This module defines strict, reusable prompt templates for AI feedback generation.

KEY PRINCIPLES:
1. Prompts are deterministic and controlled
2. AI receives structured analysis (NOT raw resume)
3. AI provides explanations (NOT decisions)
4. Templates clearly instruct AI to NOT calculate scores or make decisions
"""

from typing import Dict, List, Any, Optional
from string import Template


class PromptTemplateManager:
    """
    Manages prompt templates for different types of AI feedback.
    
    Templates are designed to:
    - Receive structured data (scores, skills, etc.)
    - Generate human-readable explanations
    - Provide actionable suggestions
    - Never make scoring or matching decisions
    """
    
    # System prompt that sets the AI's role
    SYSTEM_PROMPT = """You are an expert resume advisor and career coach.

Your role is to provide clear, actionable feedback to help candidates improve their resumes.

CRITICAL RULES:
1. You do NOT calculate scores or ratings
2. You do NOT make hiring decisions
3. You do NOT determine candidate eligibility
4. You ONLY provide explanations and suggestions based on the analysis already performed

You receive structured analysis data and provide human-readable feedback."""
    
    # Template for comprehensive feedback
    COMPREHENSIVE_FEEDBACK_TEMPLATE = Template("""
Analyze this resume performance for a ${job_role} position:

FINAL SCORE: ${final_score}/100 (${interpretation})

SCORE BREAKDOWN:
- Skill Match: ${skill_score}% (weight: 40%)
- Semantic Alignment: ${semantic_score}% (weight: 25%)
- Experience Match: ${experience_score}% (weight: 20%)
- Project Relevance: ${project_score}% (weight: 15%)

MATCHED SKILLS (${matched_count}):
${matched_skills}

MISSING SKILLS (${missing_count}):
${missing_skills}

EXPERIENCE:
- Candidate has ${experience_years} years
- Required: ${required_experience} years
- Match: ${experience_match}

PROJECTS:
- Total projects: ${project_count}
- Relevant projects: ${relevant_projects}

Based on this analysis (NOT by making your own evaluation), provide:

1. RESUME SUMMARY (2-3 sentences):
   - Overall alignment with the role
   - Key strengths
   - Main areas for improvement

2. SKILL GAP ANALYSIS (for missing skills only):
   - Which missing skills are most critical for this role?
   - Learning path recommendations
   - Timeline suggestions

3. IMPROVEMENT SUGGESTIONS (3-5 specific actions):
   - How to highlight existing strengths better
   - What additional information to add
   - How to restructure or reframe experience

4. ATS OPTIMIZATION TIPS (3-4 technical suggestions):
   - Keyword optimization
   - Formatting recommendations
   - Section structure advice

Format your response as clear, actionable bullet points.
Be encouraging but honest.
Focus on what the candidate CAN control.
""")
    
    # Template for skill-focused feedback
    SKILL_FEEDBACK_TEMPLATE = Template("""
Provide skill development advice for a ${job_role} candidate:

MATCHED SKILLS: ${matched_skills}
MISSING SKILLS: ${missing_skills}

Current skill match: ${skill_match_percent}%

Provide:
1. Priority ranking of missing skills (most critical first)
2. Learning resources or paths for each missing skill
3. Approximate time to acquire each skill
4. How to demonstrate these skills on a resume

Be specific and actionable.
""")
    
    # Template for experience-focused feedback
    EXPERIENCE_FEEDBACK_TEMPLATE = Template("""
Provide experience optimization advice:

CANDIDATE EXPERIENCE: ${experience_years} years
REQUIRED EXPERIENCE: ${required_experience} years
MATCH STATUS: ${experience_match}

PROJECT COUNT: ${project_count}
RELEVANT PROJECTS: ${relevant_projects}

Provide:
1. How to better quantify and showcase existing experience
2. What types of projects to highlight
3. How to frame responsibilities for maximum impact
4. Suggestions for filling experience gaps

Be practical and encouraging.
""")
    
    # Template for ATS optimization
    ATS_OPTIMIZATION_TEMPLATE = Template("""
Provide ATS (Applicant Tracking System) optimization advice:

JOB ROLE: ${job_role}
CURRENT SCORE: ${final_score}/100

KEY KEYWORDS PRESENT: ${matched_skills}
MISSING KEYWORDS: ${missing_skills}

Provide specific ATS optimization tips:
1. Keyword placement strategies
2. Section naming conventions
3. Formatting do's and don'ts
4. How to increase ATS score without misleading

Be technical and specific.
""")
    
    # Template for concise feedback (faster response)
    CONCISE_FEEDBACK_TEMPLATE = Template("""
Provide brief, actionable feedback for a ${job_role} candidate:

Score: ${final_score}/100 (${interpretation})
Missing Skills: ${missing_skills}

Provide:
1. One-sentence summary
2. Top 3 improvement actions
3. Most critical skill to learn

Be concise and direct.
""")
    
    def __init__(self):
        """Initialize the prompt template manager"""
        self.templates = {
            'comprehensive': self.COMPREHENSIVE_FEEDBACK_TEMPLATE,
            'skill_focused': self.SKILL_FEEDBACK_TEMPLATE,
            'experience_focused': self.EXPERIENCE_FEEDBACK_TEMPLATE,
            'ats_optimization': self.ATS_OPTIMIZATION_TEMPLATE,
            'concise': self.CONCISE_FEEDBACK_TEMPLATE,
        }
    
    def generate_comprehensive_prompt(
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
    ) -> str:
        """
        Generate comprehensive feedback prompt with all analysis data.
        
        Args:
            job_role: Target job role
            final_score: Final resume score (0-100)
            interpretation: Score interpretation (Excellent/Good/etc.)
            score_breakdown: Dict with skill_score, semantic_score, etc.
            matched_skills: List of matched skills
            missing_skills: List of missing skills
            experience_years: Candidate's years of experience
            required_experience: Required years of experience
            experience_match: Whether experience requirement is met
            project_count: Total number of projects
            relevant_projects: Number of relevant projects
        
        Returns:
            Formatted prompt string
        """
        return self.COMPREHENSIVE_FEEDBACK_TEMPLATE.substitute(
            job_role=job_role,
            final_score=final_score,
            interpretation=interpretation,
            skill_score=int(score_breakdown.get('skill_match', 0) * 100 / 40),  # Denormalize
            semantic_score=int(score_breakdown.get('semantic_similarity', 0) * 100 / 25),
            experience_score=int(score_breakdown.get('experience', 0) * 100 / 20),
            project_score=int(score_breakdown.get('project_score', 0) * 100 / 15),
            matched_count=len(matched_skills),
            matched_skills=', '.join(matched_skills) if matched_skills else 'None',
            missing_count=len(missing_skills),
            missing_skills=', '.join(missing_skills) if missing_skills else 'None',
            experience_years=experience_years,
            required_experience=required_experience,
            experience_match='Yes' if experience_match else 'No',
            project_count=project_count,
            relevant_projects=relevant_projects
        )
    
    def generate_skill_feedback_prompt(
        self,
        job_role: str,
        matched_skills: List[str],
        missing_skills: List[str],
        skill_match_percent: float
    ) -> str:
        """
        Generate skill-focused feedback prompt.
        
        Args:
            job_role: Target job role
            matched_skills: List of matched skills
            missing_skills: List of missing skills
            skill_match_percent: Skill match percentage (0-100)
        
        Returns:
            Formatted prompt string
        """
        return self.SKILL_FEEDBACK_TEMPLATE.substitute(
            job_role=job_role,
            matched_skills=', '.join(matched_skills) if matched_skills else 'None',
            missing_skills=', '.join(missing_skills) if missing_skills else 'None',
            skill_match_percent=int(skill_match_percent)
        )
    
    def generate_experience_feedback_prompt(
        self,
        experience_years: float,
        required_experience: float,
        experience_match: bool,
        project_count: int,
        relevant_projects: int
    ) -> str:
        """
        Generate experience-focused feedback prompt.
        
        Args:
            experience_years: Candidate's years of experience
            required_experience: Required years of experience
            experience_match: Whether experience requirement is met
            project_count: Total number of projects
            relevant_projects: Number of relevant projects
        
        Returns:
            Formatted prompt string
        """
        return self.EXPERIENCE_FEEDBACK_TEMPLATE.substitute(
            experience_years=experience_years,
            required_experience=required_experience,
            experience_match='Yes' if experience_match else 'No',
            project_count=project_count,
            relevant_projects=relevant_projects
        )
    
    def generate_ats_optimization_prompt(
        self,
        job_role: str,
        final_score: int,
        matched_skills: List[str],
        missing_skills: List[str]
    ) -> str:
        """
        Generate ATS optimization feedback prompt.
        
        Args:
            job_role: Target job role
            final_score: Final resume score (0-100)
            matched_skills: List of matched skills
            missing_skills: List of missing skills
        
        Returns:
            Formatted prompt string
        """
        return self.ATS_OPTIMIZATION_TEMPLATE.substitute(
            job_role=job_role,
            final_score=final_score,
            matched_skills=', '.join(matched_skills) if matched_skills else 'None',
            missing_skills=', '.join(missing_skills) if missing_skills else 'None'
        )
    
    def generate_concise_feedback_prompt(
        self,
        job_role: str,
        final_score: int,
        interpretation: str,
        missing_skills: List[str]
    ) -> str:
        """
        Generate concise feedback prompt for quick responses.
        
        Args:
            job_role: Target job role
            final_score: Final resume score (0-100)
            interpretation: Score interpretation
            missing_skills: List of missing skills
        
        Returns:
            Formatted prompt string
        """
        return self.CONCISE_FEEDBACK_TEMPLATE.substitute(
            job_role=job_role,
            final_score=final_score,
            interpretation=interpretation,
            missing_skills=', '.join(missing_skills[:5]) if missing_skills else 'None'  # Limit to 5
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt that defines AI's role"""
        return self.SYSTEM_PROMPT
    
    def get_template(self, template_name: str) -> Optional[Template]:
        """
        Get a specific template by name.
        
        Args:
            template_name: Name of the template
        
        Returns:
            Template object or None if not found
        """
        return self.templates.get(template_name)
    
    def list_templates(self) -> List[str]:
        """Get list of available template names"""
        return list(self.templates.keys())
