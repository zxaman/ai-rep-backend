# ⚡ PHASE 5 QUICK REFERENCE

## 🎯 One-Liner
**Transparent resume scoring (0-100) with explainable weighted formula - no black-box AI!**

---

## 📊 Scoring Formula

```
final_score = (
    skill_match × 40% +
    semantic_sim × 25% +
    experience × 20% +
    projects × 15%
) × 100
```

---

## 🔌 API Endpoint

### POST `/api/calculate-score`

**Request:**
```json
{
  "normalized_features": {
    "skill_match": 0.75,
    "experience": 0.80,
    "project_score": 0.65
  },
  "semantic_similarity": 0.72,
  "include_breakdown": true
}
```

**Response:**
```json
{
  "final_score": 74,
  "interpretation": "Good",
  "emoji": "👍",
  "recommendation": "Strong candidate",
  "weighted_components": {
    "skill_match": 30.0,
    "semantic_similarity": 18.0,
    "experience": 16.0,
    "project_score": 9.75
  },
  "breakdown": {
    "visualizations": {...},
    "insights": {...}
  }
}
```

---

## 🏷️ Score Ranges

| Score | Label | Emoji | Meaning |
|-------|-------|-------|---------|
| 85-100 | Excellent | 🌟 | Exceptional match |
| 70-84 | Good | 👍 | Strong candidate |
| 55-69 | Moderate | 👌 | Decent with potential |
| 40-54 | Weak | 👎 | Significant gaps |
| 0-39 | Poor | ❌ | Not suitable |

---

## 🎨 Visualizations

1. **Pie Chart** - Component weight distribution
2. **Bar Chart** - Weighted scores comparison
3. **Radar Chart** - Multi-dimensional view
4. **Gauge Chart** - Overall score meter
5. **Progress Bars** - Individual components

---

## 🔧 Quick Setup

### 1. Start API
```bash
cd backend-api
uvicorn app.main:app --reload --port 8000
```

### 2. Test Endpoint
```bash
curl -X POST http://localhost:8000/api/calculate-score \
  -H "Content-Type: application/json" \
  -d '{
    "normalized_features": {
      "skill_match": 0.75,
      "experience": 0.80,
      "project_score": 0.65
    },
    "semantic_similarity": 0.72,
    "include_breakdown": true
  }'
```

---

## 💻 Code Usage

### Calculate Score
```python
from app.scoring import ScoringEngine

engine = ScoringEngine()

result = engine.calculate_score(
    skill_match=0.75,
    semantic_similarity=0.72,
    experience=0.80,
    project_score=0.65
)

print(f"Score: {result['final_score']}/100")
print(f"Label: {result['interpretation']} {result['emoji']}")
```

### Generate Breakdown
```python
from app.scoring import ScoreBreakdownGenerator

generator = ScoreBreakdownGenerator(
    config_weights=engine.get_weights()
)

breakdown = generator.generate_breakdown(
    score_result=result,
    normalized_features={'skill_match': 0.75, ...},
    semantic_similarity=0.72
)

print(breakdown['insights']['strengths'])
print(breakdown['insights']['weaknesses'])
```

---

## ⚙️ Configuration

**File:** `app/scoring/scoring_config.json`

### Change Weights
```json
{
  "weights": {
    "skill_match": 0.50,         // Increase to 50%
    "semantic_similarity": 0.20, // Decrease to 20%
    "experience": 0.20,
    "project_score": 0.10
  }
}
```
**Note:** Must sum to 1.0

### Adjust Thresholds
```json
{
  "interpretation_thresholds": {
    "excellent": 90,  // Raise bar
    "good": 75,
    "moderate": 60,
    "weak": 45,
    "poor": 0
  }
}
```

---

## 🧪 Run Tests

```bash
# All tests
pytest tests/test_scoring_engine.py -v
pytest tests/test_score_breakdown.py -v
pytest tests/test_full_pipeline.py -v -s

# Specific test
pytest tests/test_scoring_engine.py::TestScoreCalculation::test_calculate_perfect_score -v
```

---

## 🎤 Interview Sound Bite

> "I built a transparent resume scoring model using a weighted formula - 40% skill match, 25% semantic similarity, 20% experience, 15% projects. Unlike black-box AI, every score is explainable. Weights are configurable via JSON. I wrote 65+ tests and it generates Chart.js-ready visualizations. It's exactly the kind of interpretable system interviewers want to see!"

---

## 📁 Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `app/scoring/scoring_engine.py` | Calculate final score | ~400 |
| `app/scoring/score_breakdown.py` | Generate visualizations | ~400 |
| `app/scoring/scoring_config.json` | Configuration | 60 |
| `tests/test_scoring_engine.py` | Unit tests (engine) | ~550 |
| `tests/test_score_breakdown.py` | Unit tests (breakdown) | ~600 |
| `tests/test_full_pipeline.py` | Integration tests | ~400 |

---

## 🔗 Related Endpoints

| Endpoint | Purpose | Phase |
|----------|---------|-------|
| `POST /api/parse-resume` | Parse & preprocess | 2 |
| `POST /api/extract-features` | Feature engineering | 3 |
| `POST /api/compute-similarity` | Semantic similarity | 4 |
| `POST /api/calculate-score` | Final scoring | 5 |
| `GET /api/health` | Health check | All |

---

## 📊 Component Weights

| Component | Weight | Reasoning |
|-----------|--------|-----------|
| Skill Match | 40% | Most critical factor |
| Semantic Sim | 25% | Alignment with job description |
| Experience | 20% | Years of relevant experience |
| Projects | 15% | Demonstrable work |

---

## 🎯 Key Features

- ✅ Transparent weighted formula (no black-box)
- ✅ JSON configuration (no code changes)
- ✅ 5 visualization types (Chart.js-ready)
- ✅ Insights generation (strengths/weaknesses/recommendations)
- ✅ Input validation (0-1 range)
- ✅ 65+ comprehensive tests
- ✅ Production-ready error handling
- ✅ Full API documentation

---

## 🚀 Full Pipeline Example

```python
import requests

base_url = "http://localhost:8000"

# 1. Parse resume
with open('resume.pdf', 'rb') as f:
    parse_resp = requests.post(f"{base_url}/api/parse-resume", files={'file': f})
parsed = parse_resp.json()

# 2. Extract features
features_resp = requests.post(f"{base_url}/api/extract-features", json={
    'resume_sections': parsed['sections'],
    'job_role_data': {
        'required_skills': ['Python', 'FastAPI', 'ML'],
        'years_of_experience': 3.0
    }
})
features = features_resp.json()

# 3. Compute similarity
sim_resp = requests.post(f"{base_url}/api/compute-similarity", json={
    'resume_text': parsed['cleaned_text'],
    'job_text': 'Looking for ML Engineer...'
})
similarity = sim_resp.json()

# 4. Calculate score
score_resp = requests.post(f"{base_url}/api/calculate-score", json={
    'normalized_features': features['normalized_features'],
    'semantic_similarity': similarity['semantic_similarity_score'],
    'include_breakdown': True
})
score = score_resp.json()

print(f"Final Score: {score['final_score']}/100 - {score['interpretation']}")
```

---

## 📈 Stats

- **Lines of Code:** ~960
- **Unit Tests:** 55+
- **Integration Tests:** 10+
- **Total Tests:** 65+
- **API Endpoints:** 1 new (`/api/calculate-score`)
- **Visualizations:** 5 types
- **Configuration:** JSON-based (60 lines)

---

## 🏆 Interview Impact

| Aspect | Rating |
|--------|--------|
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Explainability | ⭐⭐⭐⭐⭐ |
| Production Readiness | ⭐⭐⭐⭐⭐ |
| Interview Appeal | ⭐⭐⭐⭐⭐ |

---

## 📚 Documentation

- **Full Guide:** [PHASE5_README.md](./PHASE5_README.md)
- **Summary:** [PHASE5_SUMMARY.md](./PHASE5_SUMMARY.md)
- **API Docs:** http://localhost:8000/docs
- **Previous Phase:** [PHASE4_README.md](./PHASE4_README.md)

---

## ✅ Checklist

- [x] Transparent scoring engine
- [x] JSON configuration
- [x] Score breakdown generator
- [x] 5 visualization types
- [x] Insights generation
- [x] API endpoint
- [x] 65+ tests
- [x] Full documentation

---

**Phase 5: Complete ✅**  
**Status: Production-Ready**  
**Interview Impact: Maximum 🎯**
