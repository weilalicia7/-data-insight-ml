# -*- coding: utf-8 -*-
import sys
import io

# Force UTF-8 encoding for stdout
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Test script for the Mentorship Risk Prediction API.

Tests both /api/health and /api/predict endpoints.

Usage:
    # Start the server first:
    python app.py

    # Then in another terminal:
    python test_api.py
"""

import requests
import json
from datetime import datetime

# API Configuration
BASE_URL = 'http://localhost:5000'
HEALTH_ENDPOINT = f'{BASE_URL}/api/health'
PREDICT_ENDPOINT = f'{BASE_URL}/api/predict'


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(response, description):
    """Print formatted API response."""
    print(f"\n{description}")
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))


def test_health_check():
    """Test the /api/health endpoint."""
    print_header("TEST 1: Health Check")

    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        print_result(response, "✓ Health check successful")

        if response.status_code == 200:
            data = response.json()
            print(f"\nModel Status: {data.get('model')}")
            print(f"Scaler Status: {data.get('scaler')}")
            print(f"Features: {data.get('features')}")
            return True
        else:
            print("✗ Health check failed")
            return False

    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Cannot connect to server.")
        print("Make sure the Flask server is running: python app.py")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        return False


def test_prediction(test_case):
    """Test the /api/predict endpoint with a test case."""
    name = test_case['name']
    payload = test_case['data']

    print_header(f"TEST: {name}")

    try:
        response = requests.post(
            PREDICT_ENDPOINT,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        print_result(response, f"Prediction result for: {name}")

        if response.status_code == 200:
            data = response.json()
            prediction = data.get('prediction', {})

            print("\n" + "-" * 70)
            print("RISK METRICS:")
            print("-" * 70)
            print(f"Response Risk:      {prediction.get('responseRisk')}%")
            print(f"Match Quality:      {prediction.get('matchQuality')}%")
            print(f"Motivation Risk:    {prediction.get('motivationRisk')}%")
            print(f"Days to Failure:    {prediction.get('daysToFailure')} days")
            print(f"Model Used:         {data.get('model')}")
            print("-" * 70)

            # Risk assessment
            risk = prediction.get('responseRisk', 0)
            if risk >= 80:
                risk_level = "🔴 CRITICAL RISK"
            elif risk >= 60:
                risk_level = "🟠 HIGH RISK"
            elif risk >= 40:
                risk_level = "🟡 MEDIUM RISK"
            elif risk >= 20:
                risk_level = "🟢 LOW RISK"
            else:
                risk_level = "✅ VERY LOW RISK"

            print(f"\nRisk Level: {risk_level}")
            return True

        else:
            print(f"\n✗ Prediction failed with status {response.status_code}")
            return False

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        return False


def run_all_tests():
    """Run all test cases."""

    print("\n" + "=" * 70)
    print("  MENTORSHIP RISK PREDICTION API - TEST SUITE")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # Test 1: Health Check
    health_ok = test_health_check()

    if not health_ok:
        print("\n✗ Server is not responding. Stopping tests.")
        return

    # Define test cases
    test_cases = [
        {
            'name': 'High Risk Case (Computer Science, Low Engagement, Summer)',
            'data': {
                'workfield': 'Computer Science',
                'study_level': 'Bac+1',
                'needs': 'Professional',
                'registration_month': 'July',
                'engagement_score': 0.5,
                'project_confidence_level': 2,
                'mentor_availability': 3,
                'previous_rejection': 1
            }
        },
        {
            'name': 'Low Risk Case (Teaching, High Engagement, Good Month)',
            'data': {
                'workfield': 'Teaching',
                'study_level': 'Bac+5+',
                'needs': 'Both',
                'registration_month': 'November',
                'engagement_score': 2.8,
                'project_confidence_level': 5,
                'mentor_availability': 12,
                'previous_rejection': 0
            }
        },
        {
            'name': 'Medium Risk Case (Engineering, Average Engagement)',
            'data': {
                'workfield': 'Engineering',
                'study_level': 'Bac+3',
                'needs': 'Academic',
                'registration_month': 'March',
                'engagement_score': 1.5,
                'project_confidence_level': 3,
                'mentor_availability': 6,
                'previous_rejection': 0
            }
        },
        {
            'name': 'Edge Case (Very Low Engagement, Multiple Risk Factors)',
            'data': {
                'workfield': 'Computer Science',
                'study_level': 'Bac+1',
                'needs': 'Professional',
                'registration_month': 'June',
                'engagement_score': 0.2,
                'project_confidence_level': 1,
                'mentor_availability': 1,
                'previous_rejection': 1
            }
        },
        {
            'name': 'Optimal Case (All Positive Indicators)',
            'data': {
                'workfield': 'Healthcare',
                'study_level': 'Bac+4',
                'needs': 'Both',
                'registration_month': 'January',
                'engagement_score': 3.0,
                'project_confidence_level': 5,
                'mentor_availability': 15,
                'previous_rejection': 0
            }
        }
    ]

    # Run all test cases
    results = []
    for test_case in test_cases:
        result = test_prediction(test_case)
        results.append(result)

    # Summary
    print_header("TEST SUMMARY")
    total_tests = len(test_cases) + 1  # +1 for health check
    passed_tests = sum(results) + (1 if health_ok else 0)

    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")

    if passed_tests == total_tests:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n⚠ {total_tests - passed_tests} test(s) failed")

    print("=" * 70 + "\n")


def test_error_handling():
    """Test API error handling with invalid inputs."""
    print_header("TEST: Error Handling")

    # Test 1: Missing fields
    print("\n--- Test 1: Missing required fields ---")
    try:
        response = requests.post(
            PREDICT_ENDPOINT,
            json={'workfield': 'Engineering'},  # Missing other fields
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Invalid JSON
    print("\n--- Test 2: Invalid JSON ---")
    try:
        response = requests.post(
            PREDICT_ENDPOINT,
            data="not json",  # Invalid JSON
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: Invalid endpoint
    print("\n--- Test 3: Invalid endpoint ---")
    try:
        response = requests.get(f'{BASE_URL}/api/invalid')
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    # Run main test suite
    run_all_tests()

    # Optional: Test error handling
    test_error_handling()

    print("\n✓ Testing complete!")
