"""
Manual API Test Script for Phase 5 Scoring Endpoint

This script tests the /api/calculate-score endpoint with sample data
to verify the scoring system is working correctly.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test API health endpoint"""
    print("\n" + "="*60)
    print("1. TESTING HEALTH CHECK")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/health")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Health check passed")
        print(f"Status: {data['status']}")
        print(f"Components: {json.dumps(data['components'], indent=2)}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False


def test_root_endpoint():
    """Test root endpoint to verify Phase 5"""
    print("\n" + "="*60)
    print("2. TESTING ROOT ENDPOINT")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Root endpoint passed")
        print(f"Phase: {data['phase']}")
        print(f"Version: {data['version']}")
        return True
    else:
        print(f"❌ Root endpoint failed: {response.status_code}")
        return False


def test_calculate_score_high_performer():
    """Test scoring with high-performing candidate"""
    print("\n" + "="*60)
    print("3. TESTING CALCULATE SCORE - HIGH PERFORMER")
    print("="*60)
    
    request_data = {
        "normalized_features": {
            "skill_match": 0.85,      # 85% skill match
            "experience": 0.90,       # 90% experience match
            "project_score": 0.80     # 80% project relevance
        },
        "semantic_similarity": 0.88,  # 88% semantic alignment
        "include_breakdown": True
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/api/calculate-score",
        json=request_data
    )
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ Score calculation passed")
        print(f"\nFinal Score: {data['final_score']}/100")
        print(f"Interpretation: {data['interpretation']} {data['emoji']}")
        print(f"Recommendation: {data['recommendation']}")
        
        print(f"\nWeighted Components:")
        for component, value in data['weighted_components'].items():
            print(f"  - {component}: {value:.2f}")
        
        if 'breakdown' in data:
            print(f"\nBreakdown Included:")
            print(f"  - Visualizations: {list(data['breakdown']['visualizations'].keys())}")
            print(f"  - Strengths: {len(data['breakdown']['insights']['strengths'])}")
            print(f"  - Weaknesses: {len(data['breakdown']['insights']['weaknesses'])}")
            print(f"  - Recommendations: {len(data['breakdown']['insights']['recommendations'])}")
        
        return True
    else:
        print(f"❌ Score calculation failed: {response.status_code}")
        print(f"Error: {response.text}")
        return False


def test_calculate_score_average_candidate():
    """Test scoring with average candidate"""
    print("\n" + "="*60)
    print("4. TESTING CALCULATE SCORE - AVERAGE CANDIDATE")
    print("="*60)
    
    request_data = {
        "normalized_features": {
            "skill_match": 0.60,      # 60% skill match
            "experience": 0.65,       # 65% experience match
            "project_score": 0.55     # 55% project relevance
        },
        "semantic_similarity": 0.62,  # 62% semantic alignment
        "include_breakdown": True
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/api/calculate-score",
        json=request_data
    )
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ Score calculation passed")
        print(f"\nFinal Score: {data['final_score']}/100")
        print(f"Interpretation: {data['interpretation']} {data['emoji']}")
        print(f"Recommendation: {data['recommendation']}")
        return True
    else:
        print(f"❌ Score calculation failed: {response.status_code}")
        return False


def test_calculate_score_low_performer():
    """Test scoring with low-performing candidate"""
    print("\n" + "="*60)
    print("5. TESTING CALCULATE SCORE - LOW PERFORMER")
    print("="*60)
    
    request_data = {
        "normalized_features": {
            "skill_match": 0.30,      # 30% skill match
            "experience": 0.35,       # 35% experience match
            "project_score": 0.25     # 25% project relevance
        },
        "semantic_similarity": 0.32,  # 32% semantic alignment
        "include_breakdown": True
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/api/calculate-score",
        json=request_data
    )
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ Score calculation passed")
        print(f"\nFinal Score: {data['final_score']}/100")
        print(f"Interpretation: {data['interpretation']} {data['emoji']}")
        print(f"Recommendation: {data['recommendation']}")
        return True
    else:
        print(f"❌ Score calculation failed: {response.status_code}")
        return False


def test_calculate_score_without_breakdown():
    """Test scoring without breakdown (faster response)"""
    print("\n" + "="*60)
    print("6. TESTING CALCULATE SCORE - WITHOUT BREAKDOWN")
    print("="*60)
    
    request_data = {
        "normalized_features": {
            "skill_match": 0.75,
            "experience": 0.80,
            "project_score": 0.70
        },
        "semantic_similarity": 0.75,
        "include_breakdown": False  # No breakdown
    }
    
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/api/calculate-score",
        json=request_data
    )
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ Score calculation passed")
        print(f"\nFinal Score: {data['final_score']}/100")
        print(f"Interpretation: {data['interpretation']} {data['emoji']}")
        
        if data.get('breakdown') is None:
            print("✅ Breakdown correctly excluded (faster response)")
        else:
            print("⚠️  Breakdown included despite include_breakdown=False")
        
        return True
    else:
        print(f"❌ Score calculation failed: {response.status_code}")
        return False


def test_invalid_input_validation():
    """Test input validation with invalid data"""
    print("\n" + "="*60)
    print("7. TESTING INPUT VALIDATION - INVALID DATA")
    print("="*60)
    
    request_data = {
        "normalized_features": {
            "skill_match": 1.5,       # Invalid: > 1.0
            "experience": 0.80,
            "project_score": 0.70
        },
        "semantic_similarity": 0.75,
        "include_breakdown": False
    }
    
    print(f"Request (with invalid skill_match=1.5): {json.dumps(request_data, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/api/calculate-score",
        json=request_data
    )
    
    if response.status_code in [400, 422, 500]:
        print(f"✅ Validation correctly rejected invalid input (status {response.status_code})")
        return True
    else:
        print(f"⚠️  Invalid input accepted (status {response.status_code})")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PHASE 5 API MANUAL TESTING")
    print("="*60)
    
    results = []
    
    # Run all tests
    results.append(("Health Check", test_health_check()))
    results.append(("Root Endpoint", test_root_endpoint()))
    results.append(("High Performer Score", test_calculate_score_high_performer()))
    results.append(("Average Candidate Score", test_calculate_score_average_candidate()))
    results.append(("Low Performer Score", test_calculate_score_low_performer()))
    results.append(("Score Without Breakdown", test_calculate_score_without_breakdown()))
    results.append(("Input Validation", test_invalid_input_validation()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 5 API is working correctly!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API server")
        print("Make sure the API is running: uvicorn app.main:app --reload --port 8000")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
