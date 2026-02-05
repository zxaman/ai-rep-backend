# Phase 2: Resume Parsing & NLP Preprocessing

## Overview
This phase implements the backend API foundation for resume text extraction, NLP preprocessing, and section segmentation. **No AI/LLM integration, skill matching, or scoring logic is included in Phase 2.**

## Architecture

### Services Layer
```
app/services/
├── resume_parser.py       # PDF/DOCX text extraction
├── text_preprocessor.py   # NLP cleaning & tokenization
└── section_extractor.py   # Resume section segmentation
```

### API Endpoint
```
POST /api/parse-resume
- Accepts: PDF or DOCX file (max 10MB)
- Returns: Raw text, cleaned tokens, extracted sections, metadata
```

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Web Framework | FastAPI | 0.109.0 |
| ASGI Server | Uvicorn | 0.27.0 |
| PDF Parsing | pdfplumber | 0.10.3 |
| DOCX Parsing | python-docx | 1.1.0 |
| NLP Engine | spaCy | 3.7.2 |
| Language Model | en_core_web_sm | 3.7.0 |
| Testing | pytest | 7.4.4 |

## Installation

### 1. Create Virtual Environment
```bash
cd "C:\Users\Admin\Desktop\sakshi project part 2\backend-api"
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```

### 4. Configure Environment
```bash
copy .env.example .env
# Edit .env if needed (default settings work for development)
```

## Running the API

### Development Mode (with auto-reload)
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Alternative: Using Python directly
```bash
python -m app.main
```

### API will be available at:
- **Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Service Details

### 1. Resume Parser Service (`resume_parser.py`)
**Purpose**: Extract raw text from resume files

**Supported Formats**:
- PDF (via pdfplumber)
- DOCX/DOC (via python-docx)

**Methods**:
- `parse(file_content, filename) -> str`: Extract text from file
- `validate_file(filename) -> bool`: Check if file type is supported

**Features**:
- Multi-page PDF support
- DOCX table extraction
- Error handling for corrupted files

### 2. Text Preprocessor Service (`text_preprocessor.py`)
**Purpose**: Clean and tokenize text using NLP

**Pipeline Steps**:
1. Lowercase conversion
2. Special character removal (keeps +, #, -, / for skills like C++, C#)
3. Tokenization with spaCy
4. Stopword removal
5. Lemmatization

**Methods**:
- `preprocess(text) -> List[str]`: Full preprocessing pipeline
- `clean_text_only(text) -> str`: Clean without tokenization
- `extract_keywords(tokens, top_n) -> List[str]`: Get most frequent tokens

**Output Example**:
```python
Input: "Python Developer with 5 years of experience in Machine Learning."
Output: ['python', 'developer', 'year', 'experience', 'machine', 'learning']
```

### 3. Section Extractor Service (`section_extractor.py`)
**Purpose**: Identify and extract resume sections using regex patterns

**Detected Sections**:
- Skills / Technical Skills / Core Competencies
- Experience / Work History / Employment History
- Projects / Key Projects
- Education / Academic Background
- Certifications / Licenses / Credentials
- Summary / Profile / Objective

**Methods**:
- `extract_sections(text) -> Dict[str, str]`: Extract all sections
- `get_section(text, section_name) -> str`: Get specific section
- `list_found_sections(text) -> List[str]`: List detected sections

**Approach**:
- Regex pattern matching for section headers
- Heuristic-based header detection
- Line-based segmentation

## API Usage

### Health Check
```bash
curl http://localhost:8000/
```

**Response**:
```json
{
  "status": "online",
  "service": "AI-Powered Resume Analyzer API",
  "phase": "Phase 2: NLP Foundation",
  "version": "0.2.0"
}
```

### Parse Resume
```bash
curl -X POST "http://localhost:8000/api/parse-resume" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_resume.pdf"
```

**Response**:
```json
{
  "raw_text": "John Doe\nSenior Machine Learning Engineer...",
  "cleaned_tokens": ["john", "doe", "senior", "machine", "learning", "engineer", ...],
  "sections": {
    "skills": "Python, TensorFlow, PyTorch, AWS...",
    "experience": "Senior ML Engineer - Tech Corp (2021-Present)...",
    "projects": "Built NLP-based sentiment analysis system...",
    "education": "MSc Computer Science, Stanford University, 2020",
    "certifications": "AWS Certified Machine Learning Specialist"
  },
  "metadata": {
    "filename": "sample_resume.pdf",
    "file_size_bytes": 245678,
    "file_type": "application/pdf",
    "raw_text_length": 3456,
    "token_count": 487,
    "sections_found": ["skills", "experience", "projects", "education", "certifications"]
  }
}
```

## Testing

### Run All Tests
```bash
pytest
```

### Run with Coverage
```bash
pytest --cov=app --cov-report=html
```

### Run Specific Test File
```bash
pytest tests/test_resume_parser.py -v
```

### Run Tests by Marker
```bash
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
```

### Test Files
- `test_resume_parser.py`: Parser validation and file type tests
- `test_text_preprocessor.py`: NLP preprocessing and tokenization tests
- `test_section_extractor.py`: Section detection and extraction tests
- `conftest.py`: Shared fixtures and test configuration

## Error Handling

### File Type Validation
- **400 Bad Request**: Invalid file type (not PDF/DOCX)
- **400 Bad Request**: File size exceeds 10MB
- **400 Bad Request**: Empty or corrupted file

### Processing Errors
- **500 Internal Server Error**: PDF parsing failure
- **500 Internal Server Error**: DOCX parsing failure
- **500 Internal Server Error**: NLP model not found (run `python -m spacy download en_core_web_sm`)

## CORS Configuration

Default CORS origins (configured in `.env`):
```
CORS_ORIGINS=http://localhost:4200,http://localhost:5200
```

This allows Angular frontend (port 4200) to make API requests.

## Project Structure
```
backend-api/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   └── services/
│       ├── __init__.py
│       ├── resume_parser.py
│       ├── text_preprocessor.py
│       └── section_extractor.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_resume_parser.py
│   ├── test_text_preprocessor.py
│   └── test_section_extractor.py
├── requirements.txt
├── .env.example
├── .gitignore
└── PHASE2_README.md
```

## What's NOT in Phase 2

This phase explicitly **excludes**:
- ❌ Skill matching algorithms
- ❌ Job description analysis
- ❌ Scoring/grading logic
- ❌ Semantic similarity calculations
- ❌ AI/LLM integration
- ❌ Embeddings generation
- ❌ Frontend API integration
- ❌ Database storage

**These features will be added in Phase 3: Scoring & Matching Engine.**

## Next Steps (Phase 3)

Planned features for next phase:
1. Job description parsing and preprocessing
2. Skill extraction and matching algorithms
3. Semantic similarity using sentence embeddings
4. Experience and project relevance scoring
5. Final score calculation (40% skills + 25% similarity + 20% experience + 15% projects)
6. Frontend-backend integration

## Troubleshooting

### spaCy Model Not Found
```bash
# Error: Can't find model 'en_core_web_sm'
python -m spacy download en_core_web_sm
```

### Import Errors
```bash
# Ensure virtual environment is activated
venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### CORS Errors from Frontend
Update `.env` file:
```
CORS_ORIGINS=http://localhost:4200,http://localhost:5200
```

### Port Already in Use
Change port in `.env` or command:
```bash
uvicorn app.main:app --reload --port 8001
```

## Performance Notes

- **PDF Parsing**: ~200-500ms for typical resume (1-3 pages)
- **NLP Preprocessing**: ~100-300ms for 500-1000 tokens
- **Section Extraction**: ~50-100ms
- **Total Pipeline**: ~500ms average for complete processing

## Development Guidelines

### Adding New Section Patterns
Edit `section_extractor.py`:
```python
self.section_patterns = {
    'new_section': [
        r'\bnew\s+section\s+name\b',
        r'\balternative\s+pattern\b',
    ],
}
```

### Customizing NLP Pipeline
Edit `text_preprocessor.py`:
```python
# Enable/disable spaCy components
self.nlp.disable_pipes(["parser", "ner"])  # Disable for performance
self.nlp.enable_pipes(["parser"])          # Enable if needed
```

### Adjusting File Size Limits
Edit `.env`:
```
MAX_FILE_SIZE_MB=15  # Increase to 15MB
```

## Support

For issues or questions about Phase 2 implementation:
1. Check this README documentation
2. Review test files for usage examples
3. Check FastAPI interactive docs at `/docs` endpoint
4. Review service docstrings in source code

---

**Phase 2 Status**: ✅ Complete  
**Last Updated**: 2025  
**Next Phase**: Phase 3 - Scoring & Matching Engine
