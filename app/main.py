"""
FastAPI Main Application
Phase 2: Resume Parsing & Text Preprocessing API
Phase 3: Feature Engineering & Structured Representation API
Phase 4: Semantic Similarity Model (ML Core)
Phase 5: Custom Resume Scoring Model (Interpretability Layer)
Phase 6: AI Feedback & Recommendation Layer (Responsible LLM Integration)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv

from app.services import (
    ResumeParserService,
    TextPreprocessorService,
    SectionExtractorService,
)
from app.features import FeatureBuilder
from app.models import SimilarityEngine
from app.scoring import ScoringEngine, ScoreBreakdownGenerator
from app.ai import FeedbackGenerator

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Resume Analyzer API",
    description="NLP-based resume parsing, preprocessing, feature engineering, semantic similarity, scoring, and AI feedback service",
    version="0.6.0",
)

# CORS Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:4200").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services
resume_parser = ResumeParserService()
text_preprocessor = TextPreprocessorService()
section_extractor = SectionExtractorService()

# Initialize Feature Engineering (Phase 3)
feature_builder = FeatureBuilder()

# Initialize Similarity Engine (Phase 4)
similarity_engine = SimilarityEngine(
    tfidf_weight=0.3,
    embedding_weight=0.7
)

# Initialize Scoring Engine (Phase 5)
scoring_engine = ScoringEngine()
score_breakdown_generator = ScoreBreakdownGenerator(
    config_weights=scoring_engine.get_weights()
)

# Initialize AI Feedback Generator (Phase 6)
try:
    feedback_generator = FeedbackGenerator()
    ai_feedback_enabled = True
except Exception as e:
    print(f"Warning: AI feedback disabled - {str(e)}")
    feedback_generator = None
    ai_feedback_enabled = False

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Request Models
class JobRoleData(BaseModel):
    """Job role requirements for feature extraction"""
    required_skills: List[str] = Field(..., description="List of required skills")
    years_of_experience: float = Field(0.0, description="Required years of experience")
    required_degree: Optional[str] = Field(None, description="Required degree level")


class FeatureExtractionRequest(BaseModel):
    """Request model for feature extraction"""
    resume_sections: Dict[str, str] = Field(..., description="Resume sections from parse-resume endpoint")
    job_role_data: JobRoleData = Field(..., description="Job role requirements")


class SimilarityComputeRequest(BaseModel):
    """Request model for semantic similarity computation"""
    resume_text: str = Field(..., description="Cleaned resume text (from parse-resume endpoint)")
    job_text: str = Field(..., description="Job description text or role template")
    include_details: bool = Field(default=True, description="Include detailed breakdown from both models")


class ScoreCalculationRequest(BaseModel):
    """Request model for final resume score calculation"""
    normalized_features: Dict[str, float] = Field(..., description="Normalized features from extract-features endpoint")
    semantic_similarity: float = Field(..., description="Semantic similarity score from compute-similarity endpoint", ge=0.0, le=1.0)
    include_breakdown: bool = Field(default=True, description="Include detailed score breakdown")


# Response Models
class ParsedResumeResponse(BaseModel):
    """Response model for parsed resume"""
    cleaned_tokens: List[str]
    sections: Dict[str, str]
    metadata: Dict[str, Any]


class ExtractedFeaturesResponse(BaseModel):
    """Response model for extracted features"""
    skill_match_percent: float
    matched_skills: List[str]
    missing_skills: List[str]
    experience_years: float
    experience_match: bool
    project_count: int
    relevant_projects: int
    project_score: float
    highest_degree: Optional[str]
    education_match: bool
    normalized_features: Dict[str, float]
    metadata: Dict[str, Any]


class SimilarityComputeResponse(BaseModel):
    """Response model for semantic similarity computation"""
    semantic_similarity_score: float = Field(..., description="Combined similarity score (0-1)")
    tfidf_similarity: float = Field(..., description="TF-IDF lexical similarity (0-1)")
    embedding_similarity: float = Field(..., description="Embedding semantic similarity (0-1)")
    tfidf_weight: float = Field(..., description="Weight applied to TF-IDF score")
    embedding_weight: float = Field(..., description="Weight applied to embedding score")


class ScoreCalculationResponse(BaseModel):
    """Response model for final resume score"""
    final_score: int = Field(..., description="Final resume score (0-100)")
    interpretation: str = Field(..., description="Score interpretation (Excellent/Good/Moderate/Weak/Poor)")
    emoji: str = Field(..., description="Visual emoji representing score")
    recommendation: str = Field(..., description="Brief recommendation based on score")
    weighted_components: Dict[str, float] = Field(..., description="Breakdown of weighted components")
    breakdown: Optional[Dict[str, Any]] = Field(None, description="Detailed score breakdown with visualizations")


class FeedbackGenerationRequest(BaseModel):
    """Request model for AI feedback generation"""
    job_role: str = Field(..., description="Target job role name")
    final_score: int = Field(..., description="Final resume score from calculate-score endpoint", ge=0, le=100)
    interpretation: str = Field(..., description="Score interpretation (Excellent/Good/Moderate/Weak/Poor)")
    score_breakdown: Dict[str, float] = Field(..., description="Component score breakdown")
    matched_skills: List[str] = Field(..., description="List of matched skills")
    missing_skills: List[str] = Field(..., description="List of missing skills")
    experience_years: float = Field(..., description="Candidate's years of experience")
    required_experience: float = Field(..., description="Required years of experience")
    experience_match: bool = Field(..., description="Whether experience requirement is met")
    project_count: int = Field(..., description="Total number of projects")
    relevant_projects: int = Field(..., description="Number of relevant projects")
    feedback_type: str = Field(default="comprehensive", description="Type of feedback: comprehensive, concise, skill_focused, ats_optimization")


class FeedbackGenerationResponse(BaseModel):
    """Response model for AI-generated feedback"""
    success: bool = Field(..., description="Whether feedback was generated successfully")
    feedback_type: str = Field(..., description="Type of feedback generated")
    ai_enabled: bool = Field(..., description="Whether AI feedback is enabled")
    data: Dict[str, Any] = Field(..., description="Generated feedback data")


# Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "AI-Powered Resume Analyzer API",
        "phase": "Phase 6: AI Feedback & Recommendation Layer (Responsible LLM Integration)",
        "version": "0.6.0",
        "features": {
            "phase_2": "Resume Parsing & Text Preprocessing",
            "phase_3": "Feature Engineering & Structured Representation",
            "phase_4": "Semantic Similarity Model (ML Core)",
            "phase_5": "Custom Resume Scoring Model (Explainable)",
            "phase_6": "AI Feedback & Recommendations (Responsible LLM)"
        },
        "ai_feedback_enabled": ai_feedback_enabled
    }


@app.post("/api/parse-resume", response_model=ParsedResumeResponse)
async def parse_resume(file: UploadFile = File(...)):
    """
    Parse and preprocess uploaded resume file
    
    Steps:
    1. Extract raw text from PDF/DOCX
    2. Clean and tokenize text with NLP
    3. Extract resume sections (Skills, Experience, Projects, etc.)
    
    Args:
        file: Resume file (PDF or DOCX, max 10MB)
    
    Returns:
        ParsedResumeResponse with raw_text, cleaned_tokens, sections, metadata
    """
    
    # Validate file type
    allowed_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: PDF, DOCX. Received: {file.content_type}",
        )
    
    # Read file content
    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
    
    # Check file size
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed ({MAX_FILE_SIZE // (1024 * 1024)}MB)",
        )
    
    try:
        # Step 1: Extract raw text
        raw_text = resume_parser.parse(file_content, file.filename)
        
        if not raw_text or not raw_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from file. File may be empty or corrupted.",
            )
        
        # Step 2: Preprocess text (clean, tokenize, lemmatize)
        cleaned_tokens = text_preprocessor.preprocess(raw_text)
        
        # Step 3: Extract sections
        sections = section_extractor.extract_sections(raw_text)
        
        # Metadata
        metadata = {
            "filename": file.filename,
            "file_size_bytes": len(file_content),
            "file_type": file.content_type,
            "raw_text_length": len(raw_text),
            "token_count": len(cleaned_tokens),
            "sections_found": list(sections.keys()),
        }
        
        return ParsedResumeResponse(
            raw_text=raw_text,
            cleaned_tokens=cleaned_tokens,
            sections=sections,
            metadata=metadata,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing resume: {str(e)}",
        )


# Run with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    4. Education level detection and matching
    5. Feature normalization (0-1 range)
    
    Args:
        request: FeatureExtractionRequest with resume_sections and job_role_data
    
    Returns:
        ExtractedFeaturesResponse with all extracted and normalized features
    
    Example Request:
    ```json
    {
      "resume_sections": {
        "skills": "Python, Java, SQL, Machine Learning",
        "experience": "5 years as Software Engineer (2018-2023)",
        "projects": "Built ML model for sentiment analysis...",
        "education": "B.Tech in Computer Science"
      },
      "job_role_data": {
        "role_name": "Machine Learning Engineer",
        "required_skills": ["Python", "Machine Learning", "TensorFlow"],
        "min_experience_years": 3,
        "required_degree": "bachelor"
      }
    }
    ```
    """
    try:
        # Convert Pydantic model to dict
        job_role_dict = request.job_role_data.model_dump()
        
        # Build features using FeatureBuilder
        features = feature_builder.build_features(
            resume_sections=request.resume_sections,
            job_role_data=job_role_dict
        )
        
        # Return structured response
        return ExtractedFeaturesResponse(
            skill_match_percent=features['skill_match_percent'],
            matched_skills=features['matched_skills'],
            missing_skills=features['missing_skills'],
            experience_years=features['experience_years'],
            experience_match=features['experience_match'],
            project_count=features['project_count'],
            relevant_projects=features['relevant_projects'],
            project_score=features['project_score'],
@app.post("/api/compute-similarity", response_model=SimilarityComputeResponse)
async def compute_similarity(request: SimilarityComputeRequest = Body(...)):
    """
    Compute semantic similarity between resume and job description.
    
    This endpoint implements the ML core of the application:
    1. TF-IDF similarity (lexical/keyword matching)
    2. Sentence embedding similarity (semantic meaning)
    3. Weighted combination of both scores
    
    Args:
        request: SimilarityComputeRequest with resume_text and job_text
    
    Returns:
        SimilarityComputeResponse with combined similarity score and breakdowns
    
    Example Request:
    ```json
    {
      "resume_text": "Experienced Python developer with 5 years in ML and data science...",
      "job_text": "Looking for Machine Learning Engineer with strong Python skills...",
      "include_details": true
    }
    ```
    
    Example Response:
    ```json
    {
      "semantic_similarity_score": 0.68,
      "tfidf_similarity": 0.58,
      "embedding_similarity": 0.74,
      "tfidf_weight": 0.3,
      "embedding_weight": 0.7,
      "interpretation": "🟡 Good Match - Solid alignment with most requirements"
    }
    ```
    
    Interview Talking Point:
    "I implemented this ML pipeline combining TF-IDF for keyword matching
    and sentence transformers for semantic understanding."
    """
    try:
        # Compute similarity using the ML engine
        result = similarity_engine.compute_similarity(
            resume_text=request.resume_text,
            job_text=request.job_text,
            include_details=request.include_details
        )
        
        # Check for errors
        if "error" in result:
            raise HTTPException(
                status_code=400,
                detail=result["error"]
            )
        
        # Return structured response
        return SimilarityComputeResponse(
            semantic_similarity_score=result['semantic_similarity_score'],
            tfidf_similarity=result['tfidf_similarity'],
            embedding_similarity=result['embedding_similarity'],
            tfidf_weight=result['tfidf_weight'],
            embedding_weight=result['embedding_weight'],
            interpretation=result['interpretation'],
            tfidf_details=result.get('tfidf_details'),
            embedding_details=result.get('embedding_details')
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error computing similarity: {str(e)}"
        )


@app.post("/api/calculate-score", response_model=ScoreCalculationResponse)
async def calculate_score(request: ScoreCalculationRequest = Body(...)):
    """
    Calculate final resume score from normalized features and semantic similarity.
    
    This endpoint implements the transparent scoring model (Phase 5):
    - Combines all extracted features (Phase 3)
    - Uses semantic similarity score (Phase 4)
    - Applies weighted formula for final score (0-100)
    - Generates human-readable insights and visualizations
    
    Args:
        request: ScoreCalculationRequest with normalized_features and semantic_similarity
    
    Returns:
        ScoreCalculationResponse with final score and optional breakdown
    
    Example Request:
    ```json
    {
      "normalized_features": {
        "skill_match": 0.75,
        "experience": 0.8,
        "project_score": 0.65
      },
      "semantic_similarity": 0.72,
      "include_breakdown": true
    }
    ```
    
    Example Response:
    ```json
    {
      "final_score": 74,
      "interpretation": "Good",
      "emoji": "👍",
      "recommendation": "Strong candidate with solid qualifications",
      "weighted_components": {
        "skill_match": 30.0,
        "semantic_similarity": 18.0,
        "experience": 16.0,
        "project_score": 9.75
      },
      "breakdown": {...}
    }
    ```
    
    Interview Talking Point:
    "I designed this transparent scoring model myself instead of using a black-box AI.
    Every score component is explainable and configurable - exactly what interviewers want."
    """
    try:
        # Extract components from normalized features
        skill_match = request.normalized_features.get('skill_match', 0.0)
        experience = request.normalized_features.get('experience', 0.0)
        project_score = request.normalized_features.get('project_score', 0.0)
        
        # Calculate final score
        score_result = scoring_engine.calculate_score(
            skill_match=skill_match,
            semantic_similarity=request.semantic_similarity,
            experience=experience,
            project_score=project_score
        )
        
        # Generate breakdown if requested
        breakdown = None
        if request.include_breakdown:
            breakdown = score_breakdown_generator.generate_breakdown(
                score_result=score_result,
                normalized_features=request.normalized_features,
                semantic_similarity=request.semantic_similarity
            )
        
        # Return structured response
        return ScoreCalculationResponse(
            final_score=score_result['final_score'],
            interpretation=score_result['interpretation'],
            emoji=score_result['emoji'],
            recommendation=score_result['recommendation'],
            weighted_components=score_result['weighted_components'],
            breakdown=breakdown
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating score: {str(e)}"
        )


@app.post("/api/generate-feedback", response_model=FeedbackGenerationResponse)
async def generate_feedback(request: FeedbackGenerationRequest = Body(...)):
    """
    Generate AI-powered feedback and recommendations for resume improvement.
    
    This endpoint uses LLMs (OpenRouter/Gemini) to provide:
    - Human-readable explanations
    - Skill gap analysis
    - Improvement suggestions
    - ATS optimization tips
    
    CRITICAL: AI is used ONLY for explanations, NOT for:
    - Calculating scores
    - Making hiring decisions
    - Determining candidate eligibility
    
    All decisions are made by the data science pipeline (Phases 2-5).
    
    Args:
        request: FeedbackGenerationRequest with analysis data
    
    Returns:
        FeedbackGenerationResponse with AI-generated feedback
    
    Example Request:
    ```json
    {
      "job_role": "Data Scientist",
      "final_score": 75,
      "interpretation": "Good",
      "score_breakdown": {"skill_match": 30.0, "semantic_similarity": 18.75, ...},
      "matched_skills": ["Python", "Machine Learning"],
      "missing_skills": ["Deep Learning", "TensorFlow"],
      "experience_years": 3.5,
      "required_experience": 3.0,
      "experience_match": true,
      "project_count": 5,
      "relevant_projects": 3,
      "feedback_type": "comprehensive"
    }
    ```
    
    Example Response:
    ```json
    {
      "success": true,
      "feedback_type": "comprehensive",
      "ai_enabled": true,
      "data": {
        "ai_feedback": {
          "summary": "Your resume shows...",
          "skill_gap_analysis": "Consider learning...",
          "improvement_suggestions": ["Add more...", "Highlight..."],
          "ats_tips": ["Use keywords...", "Structure..."]
        }
      }
    }
    ```
    
    Interview Talking Point:
    "I use LLMs only for explainability and feedback, not for scoring or decisions.
    This follows responsible AI principles - the black box is only for explanations!"
    """
    try:
        # Check if AI feedback is enabled
        if not ai_feedback_enabled or feedback_generator is None:
            return FeedbackGenerationResponse(
                success=False,
                feedback_type=request.feedback_type,
                ai_enabled=False,
                data={
                    'message': 'AI feedback is not enabled. Please configure OPENROUTER_API_KEY or GEMINI_API_KEY.',
                    'fallback': 'Rule-based feedback available via scoring endpoint.'
                }
            )
        
        # Generate feedback based on type
        if request.feedback_type == "comprehensive":
            feedback = feedback_generator.generate_comprehensive_feedback(
                job_role=request.job_role,
                final_score=request.final_score,
                interpretation=request.interpretation,
                score_breakdown=request.score_breakdown,
                matched_skills=request.matched_skills,
                missing_skills=request.missing_skills,
                experience_years=request.experience_years,
                required_experience=request.required_experience,
                experience_match=request.experience_match,
                project_count=request.project_count,
                relevant_projects=request.relevant_projects
            )
        
        elif request.feedback_type == "concise":
            feedback = feedback_generator.generate_concise_feedback(
                job_role=request.job_role,
                final_score=request.final_score,
                interpretation=request.interpretation,
                missing_skills=request.missing_skills
            )
        
        elif request.feedback_type == "skill_focused":
            skill_match_percent = (request.score_breakdown.get('skill_match', 0) / 40) * 100
            feedback = feedback_generator.generate_skill_feedback(
                job_role=request.job_role,
                matched_skills=request.matched_skills,
                missing_skills=request.missing_skills,
                skill_match_percent=skill_match_percent
            )
        
        elif request.feedback_type == "ats_optimization":
            feedback = feedback_generator.generate_ats_optimization_feedback(
                job_role=request.job_role,
                final_score=request.final_score,
                matched_skills=request.matched_skills,
                missing_skills=request.missing_skills
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feedback_type: {request.feedback_type}. Must be one of: comprehensive, concise, skill_focused, ats_optimization"
            )
        
        # Return formatted response
        return FeedbackGenerationResponse(
            success='error' not in feedback,
            feedback_type=feedback.get('feedback_type', request.feedback_type),
            ai_enabled=True,
            data=feedback
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating feedback: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    """Service health check with component status"""
    return {
        "status": "healthy",
        "components": {
            "resume_parser": "ready",
            "text_preprocessor": "ready",
            "section_extractor": "ready",
            "feature_builder": "ready",
            "similarity_engine": "ready",
            "scoring_engine": "ready",
            "feedback_generator": "ready" if ai_feedback_enabled else "disabled"
        },
        "ml_models": {
            "tfidf": "scikit-learn TfidfVectorizer",
            "embeddings": "sentence-transformers/all-MiniLM-L6-v2"
        },
        "ai_feedback": {
            "enabled": ai_feedback_enabled,
            "providers": ["openrouter", "gemini"] if ai_feedback_enabled else []
        }
    }


# Run with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG_MODE", "true").lower() == "true"
    
    uvicorn.run("app.main:app", host=host, port=port, reload=debug)
