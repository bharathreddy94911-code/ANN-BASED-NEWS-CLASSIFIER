import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.json()
    assert 'status' in data
    assert 'model_available' in data

def test_dataset_info_api():
    response = client.get('/api/dataset/info')
    assert response.status_code == 200
    data = response.json()
    assert data['dataset_name'] == 'Mikami063/ag_news'
    assert data['num_classes'] == 4

def test_predict_validation_error():
    # Empty or short text should fail with 422
    response = client.post('/api/predict', json={'text': 'a'})
    assert response.status_code == 422
