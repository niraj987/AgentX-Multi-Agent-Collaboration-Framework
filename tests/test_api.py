"""Tests for APIServer endpoints."""

from fastapi.testclient import TestClient
from agentx.api.server import create_app
from agentx.memory.in_memory import InMemoryStore
from agentx.orchestrator.orchestrator import AgentOrchestrator


def test_api_status_endpoint():
    app = create_app()
    client = TestClient(app)

    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "contract_analyst" in data["registered_agents"]


def test_api_create_task_success():
    app = create_app()
    client = TestClient(app)

    payload = {
        "task_id": "api-test-task-1",
        "input_data": {
            "document_text": "Supplier shall indemnify Buyer against all patent infringement claims. Agreement terminates upon 60 days notice.",
        },
        "metadata": {
            "task_type": "full_review",
        },
    }

    response = client.post("/task", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["task_id"] == "api-test-task-1"
    assert result["status"] in ["completed", "completed_with_warnings"]
    assert "contract_analysis" in result["output_data"]

    # Verify task status endpoint
    status_resp = client.get("/status?task_id=api-test-task-1")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] in ["completed", "completed_with_warnings"]


def test_api_invalid_request_empty_task_id():
    """Verify Section 5: Invalid API request -> Return HTTP 400 with error message."""
    app = create_app()
    client = TestClient(app)

    # Missing task_id or empty task_id
    payload = {
        "task_id": "",
        "input_data": {},
        "metadata": {},
    }
    response = client.post("/task", json=payload)
    assert response.status_code == 400
    assert "Invalid API request" in response.json()["detail"]


def test_api_invalid_request_malformed_json():
    """Verify Section 5: Invalid payload schema returns HTTP 400."""
    app = create_app()
    client = TestClient(app)

    # Missing required task_id completely
    response = client.post("/task", json={"wrong_field": 123})
    assert response.status_code == 400
    assert "Invalid API request" in response.json()["detail"]
