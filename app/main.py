"""
FastAPI Main Application
Phase 2: Resume Parsing & Text Preprocessing API
Phase 3: Feature Engineering & Structured Representation API
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

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Resume Analyzer API",
    description="NLP-based resume parsing, preprocessing, and feature engineering service",
    version="0.3.0",
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

# Maximum file size (10MB)
# Request Models
class JobRoleData(BaseModel):
    """Job role requirements for feature extraction"""
    role_name: str = Field(..., description="Name of the job role")
    required_skills: List[str] = Field(default=[], description="List of required skills")
    min_experience_years: Optional[float] = Field(None, description="Minimum years of experience")
    required_degree: Optional[str] = Field(None, description="Required degree level")


class FeatureExtractionRequest(BaseModel):
    """Request model for feature extraction"""
    resume_sections: Dict[str, str] = Field(..., description="Resume sections from parse-resume endpoint")
    job_role_data: JobRoleData = Field(..., description="Job role requirements")


# Response Models
class ParsedResumeResponse(BaseModel):
    """Response model for parsed resume data"""
    raw_text: str
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
    cleaned_tokens: List[str]
    sections: Dict[str, str]
    metadata: Dict[str, any]


# Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "AI-Powered Resume Analyzer API",
        "phase": "Phase 3: Feature Engineering",
        "version": "0.3.0",
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
            highest_degree=features['highest_degree'],
            education_match=features['education_match'],
            normalized_features=features['normalized_features'],
            metadata=features['metadata']
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting features: {str(e)}"
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
        },
    }


# Run with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG_MODE", "true").lower() == "true"
    
    uvicorn.run("app.main:app", host=host, port=port, reload=debug)
