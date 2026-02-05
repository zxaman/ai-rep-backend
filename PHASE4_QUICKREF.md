# Phase 4 Quick Reference

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app.main:app --reload --port 8000

# Test endpoint
curl -X POST http://localhost:8000/api/compute-similarity \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Python ML engineer", "job_text": "ML Engineer position", "include_details": false}'
```

**API Docs**: http://localhost:8000/docs

---

## 🧩 ML Models

| Model | File | Purpose | Speed | Output |
|-------|------|---------|-------|--------|
| **TF-IDF** | `tfidf_model.py` | Keyword matching | ~50ms | Lexical similarity |
| **Embeddings** | `embedding_model.py` | Semantic meaning | ~200ms | Semantic similarity |
| **Engine** | `similarity_engine.py` | Combine both | ~250ms | Final score (0-1) |

---

## 📡 API Endpoint

### POST /api/compute-similarity

**Request**:
```json
{
  "resume_text": "5 years Python ML engineer with TensorFlow",
  "job_text": "Looking for Machine Learning Engineer with Python",
  "include_details": true
}
```

**Response**:
```json
{
  "semantic_similarity_score": 0.68,
  "tfidf_similarity": 0.58,
  "embedding_similarity": 0.74,
  "tfidf_weight": 0.3,
  "embedding_weight": 0.7,
  "interpretation": "🟡 Good Match",
  "tfidf_details": {...},
  "embedding_details": {...}
}
```

---

## 🧪 Testing

```bash
# Run all Phase 4 tests (52 total)
pytest tests/test_tfidf_model.py -v          # 13 tests
pytest tests/test_embedding_model.py -v       # 16 tests
pytest tests/test_similarity_engine.py -v     # 23 tests

# With coverage
pytest --cov=app/models --cov-report=html
```

---

## 💻 Code Usage

### Basic Similarity

```python
from app.models import SimilarityEngine

engine = SimilarityEngine()
result = engine.compute_similarity(
    resume_text="Python developer with ML",
    job_text="Machine Learning Engineer"
)

print(result['semantic_similarity_score'])  # 0.68
```

### With Custom Weights

```python
engine = SimilarityEngine(
    tfidf_weight=0.5,      # 50% keyword
    embedding_weight=0.5    # 50% semantic
)

result = engine.compute_similarity(resume, job)
```

### Batch Processing

```python
resumes = ["Resume 1", "Resume 2", "Resume 3"]
results = engine.compute_batch_similarity(resumes, job_text)

for r in results:
    print(f"Resume {r['resume_index']}: {r['semantic_similarity_score']:.2%}")
```

### Explanation

```python
explanation = engine.explain_similarity(resume_text, job_text)
print(explanation)
# Outputs detailed analysis with both model breakdowns
```

---

## 📊 Score Interpretation

| Score | Interpretation | Use Case |
|-------|----------------|----------|
| 0.75-1.0 | 🟢 Excellent Match | Strong candidate |
| 0.60-0.75 | 🟡 Good Match | Solid candidate |
| 0.45-0.60 | 🟠 Moderate Match | Possible candidate |
| 0.30-0.45 | 🔴 Weak Match | Unlikely fit |
| 0.0-0.30 | ⚫ Poor Match | Not a match |

---

## 🔧 Configuration

### Default Weights
```python
tfidf_weight = 0.3      # 30% keyword importance
embedding_weight = 0.7   # 70% semantic importance
```

### TF-IDF Parameters
```python
max_features = 500       # Top 500 terms
ngram_range = (1, 2)     # Unigrams + bigrams
stop_words = "english"   # Remove stop words
```

### Embedding Model
```python
model_name = "all-MiniLM-L6-v2"  # 384 dims, 80MB
device = None                     # Auto-detect (CPU/GPU)
```

---

## 🎯 Model Comparison

| Feature | TF-IDF | Embeddings |
|---------|--------|------------|
| **Type** | Classical NLP | Deep Learning (BERT) |
| **Matching** | Exact keywords | Semantic meaning |
| **Speed** | Very fast (~50ms) | Moderate (~200ms) |
| **Explainability** | High (shows keywords) | Low (black box) |
| **Synonyms** | ❌ Misses | ✅ Understands |
| **Context** | ❌ Weak | ✅ Strong |

---

## 🐛 Common Issues

**Import Error**: `ModuleNotFoundError: app.models`
```bash
# Restart uvicorn
uvicorn app.main:app --reload
```

**Model Download Slow**: First request downloads all-MiniLM-L6-v2 (~80MB)
```python
# Pre-download at startup
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
```

**Low Similarity Scores**: Check text is cleaned (Phase 2)
```python
# Verify with identical texts
result = engine.compute_similarity("test", "test")
assert result['semantic_similarity_score'] > 0.95
```

---

## 📁 File Locations

```
backend-api/
├── app/
│   ├── models/
│   │   ├── tfidf_model.py          # TF-IDF similarity
│   │   ├── embedding_model.py       # Sentence embeddings
│   │   └── similarity_engine.py     # Controller
│   └── main.py                      # API (v0.4.0)
├── tests/
│   ├── test_tfidf_model.py
│   ├── test_embedding_model.py
│   └── test_similarity_engine.py
├── PHASE4_README.md                 # Full docs
└── requirements.txt                 # Updated
```

---

## 🎓 Interview Soundbites

✅ **"I implemented semantic similarity using TF-IDF and sentence transformers"**  
✅ **"Combined lexical and semantic models with weighted averaging"**  
✅ **"Used all-MiniLM-L6-v2 for production-ready embeddings"**  
✅ **"Wrote 52 unit tests to validate ML model outputs"**  
✅ **"Designed for explainability with human-readable breakdowns"**

---

## 📈 Performance

| Operation | Time | Memory |
|-----------|------|--------|
| TF-IDF | 50ms | <10MB |
| Embeddings | 200ms | ~100MB (model loaded) |
| Combined | 250ms | ~100MB |
| Batch (10) | 1.5s | ~150MB |

**Optimization**: Use batch processing for multiple resumes

---

## 🔄 Integration Flow

```
Phase 2 (Parse) → Phase 4 (Similarity)
    resume_text → compute_similarity(resume_text, job_text) → 0.68

Phase 3 (Features) + Phase 4 (Similarity) → Phase 5 (Final Score)
    {features, similarity} → weighted_score → 0-100
```

---

## 📚 Key Functions

### TfidfSimilarityModel
```python
.compute_similarity(resume, job)      # Get similarity
.explain_similarity(resume, job)      # Human explanation
.get_vocabulary()                      # View extracted terms
```

### EmbeddingSimilarityModel
```python
.compute_similarity(resume, job)      # Get similarity
.compute_batch_similarity(resumes, job)  # Batch processing
.get_embedding(text)                   # Raw embedding vector
.get_model_info()                      # Model metadata
```

### SimilarityEngine
```python
.compute_similarity(resume, job, include_details=True)
.compute_batch_similarity(resumes, job)
.explain_similarity(resume, job)      # Combined explanation
.update_weights(tfidf=0.4, emb=0.6)  # Dynamic weights
.get_configuration()                   # Current settings
```

---

## ✅ Phase 4 Checklist

- [x] TF-IDF model (260 lines)
- [x] Embedding model (270 lines)
- [x] Similarity engine (310 lines)
- [x] API endpoint
- [x] 52 unit tests
- [x] Dependencies updated
- [x] Documentation complete

**Status**: ✅ COMPLETE

---

## 🚀 Next: Phase 5

**Phase 5** will combine:
- Phase 3 features (skill_match, experience, projects, education)
- Phase 4 similarity (semantic_similarity_score) ← **Now ready!**

**Formula**:
```python
final_score = (
    40% × skill_match +
    25% × semantic_similarity +  # From Phase 4
    20% × experience +
    15% × projects
)
```

---

**Version**: 0.4.0  
**Tests**: 52/52 passing ✅  
**Docs**: http://localhost:8000/docs
