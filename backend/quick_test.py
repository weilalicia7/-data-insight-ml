import requests
import json

# Test high-risk case
response = requests.post('http://localhost:5000/api/predict', json={
    'workfield': 'Computer Science',
    'study_level': 'Bac+1',
    'needs': 'Professional',
    'registration_month': 'July',
    'engagement_score': 0.5,
    'project_confidence_level': 2,
    'mentor_availability': 3,
    'previous_rejection': 1
})

print("High-Risk Case:")
print(json.dumps(response.json(), indent=2))

# Test low-risk case
response2 = requests.post('http://localhost:5000/api/predict', json={
    'workfield': 'Teaching',
    'study_level': 'Bac+5+',
    'needs': 'Both',
    'registration_month': 'November',
    'engagement_score': 2.8,
    'project_confidence_level': 5,
    'mentor_availability': 12,
    'previous_rejection': 0
})

print("\n" + "="*60)
print("Low-Risk Case:")
print(json.dumps(response2.json(), indent=2))
