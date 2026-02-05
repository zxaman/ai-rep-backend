"""
AI Module for Phase 6: Responsible LLM Integration

This module handles AI-powered feedback generation for resume analysis.

IMPORTANT: AI is used ONLY for:
- Human-readable explanations
- Improvement suggestions
- Skill gap analysis
- ATS optimization tips

AI does NOT:
- Calculate scores
- Perform matching
- Decide eligibility
- Parse resumes

All decisions are made by the data science pipeline (Phases 2-5).
"""

from app.ai.prompt_templates import PromptTemplateManager
from app.ai.ai_client import AIClient
from app.ai.feedback_generator import FeedbackGenerator

__all__ = [
    'PromptTemplateManager',
    'AIClient',
    'FeedbackGenerator',
]
