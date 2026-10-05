"""Integration tests for the FastAPI backend.

Prediction tests run against trained models when present (skipped otherwise);
everything else works on a fresh clone.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.services.model_registry import ModelRegistry

MODELS_TRAINED = all(ModelRegistry().model_status().values())
needs_models = pytest.mark.skipif(not MODELS_TRAINED, reason="trained models not present (see MANUAL_COMMANDS.md)")


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_should_report_health_and_model_availability(client: TestClient) -> None:
    body = client.get("/monitor/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert set(body["models"]) == {"wait_time", "explainer", "noshow"}


def test_should_reject_invalid_modality(client: TestClient) -> None:
    response = client.post("/predict/wait-time", json={"patient": {"patient_id": "P1", "modality": "pet"}})
    assert response.status_code == 422


@needs_models
def test_should_predict_longer_wait_when_queue_is_long(client: TestClient) -> None:
    def predict(queue: int) -> float:
        response = client.post("/predict/wait-time", json={
            "patient": {"patient_id": "P1", "modality": "mri"},
            "context": {"current_queue_length_same_modality": queue, "timestamp": "2024-07-15T10:30:00"},
            "explain": False,
        })
        assert response.status_code == 200, response.text
        return response.json()["predicted_wait_minutes"]

    assert predict(10) > predict(0)


@needs_models
def test_should_return_shap_explanation_with_clinician_text(client: TestClient) -> None:
    body = client.post("/predict/wait-time", json={"patient": {"patient_id": "P2", "modality": "ct"}}).json()
    assert body["lower_bound_minutes"] <= body["predicted_wait_minutes"] <= body["upper_bound_minutes"]
    assert body["explanation"]["contributions"]
    assert "predicted wait" in body["explanation"]["clinician_text"]


@needs_models
def test_should_return_noshow_probability_between_zero_and_one(client: TestClient) -> None:
    body = client.post("/predict/no-show", json={
        "patient": {"patient_id": "S1", "modality": "mri", "visit_type": "scheduled", "appointment_lead_time_days": 14},
    }).json()
    assert 0.0 <= body["no_show_probability"] <= 1.0


def test_should_dispatch_patient_to_free_machine_and_queue_the_next(client: TestClient) -> None:
    first = client.post("/schedule/patients", json={"patient_id": "A1", "modality": "mri"}).json()
    second = client.post("/schedule/patients", json={"patient_id": "A2", "modality": "mri"}).json()
    assert first["status"] == "in_scan"
    assert second["status"] == "waiting"
    assert second["eta"]["eta_minutes"] is not None


def test_should_preempt_routine_scan_when_emergency_arrives(client: TestClient) -> None:
    client.post("/schedule/patients", json={"patient_id": "B1", "modality": "mri"})
    body = client.post("/schedule/patients", json={"patient_id": "B2", "modality": "mri", "urgency": "emergency"}).json()
    assert body["status"] == "in_scan"
    assert body["preempted_patient_id"] == "B1"


def test_should_solve_day_ahead_schedule(client: TestClient) -> None:
    body = client.post("/schedule/day-ahead", json={"patients": [
        {"patient_id": "D1", "modality": "ct", "arrival_time_minutes": 0},
        {"patient_id": "D2", "modality": "ct", "arrival_time_minutes": 0},
        {"patient_id": "D3", "modality": "xray", "arrival_time_minutes": 30},
    ]}).json()
    assert body["status"] == "Optimal"
    assert len(body["assignments"]) == 3


def test_should_compare_policies_on_identical_streams(client: TestClient) -> None:
    body = client.post("/simulate/compare", json={
        "policies": ["fcfs", "radqueue_ai"], "days": 1, "replications": 2,
    }).json()
    served = {row["policy"]: row["patients_served"] for row in body["summary"]}
    assert served["fcfs"] == served["radqueue_ai"]  # same patients, only order differs


def test_should_return_404_for_unknown_whatif_preset(client: TestClient) -> None:
    response = client.post("/simulate/whatif", json={"scenario_name": "does_not_exist"})
    assert response.status_code == 404


def test_should_give_later_queue_positions_a_later_eta(client: TestClient) -> None:
    for pid in ("C1", "C2", "C3"):
        client.post("/schedule/patients", json={"patient_id": pid, "modality": "mri"})
    second = client.get("/schedule/patients/C2/eta").json()
    third = client.get("/schedule/patients/C3/eta").json()
    assert second["queue_position"] < third["queue_position"]
    assert second["eta_minutes"] < third["eta_minutes"]
