from __future__ import annotations

from fastapi.testclient import TestClient

from tw_stock_ai.main import app


def test_system_costs_and_effective_settings_endpoints() -> None:
    client = TestClient(app)

    costs = client.get("/api/system/costs")
    settings = client.get("/api/settings/effective")

    assert costs.status_code == 200
    assert "monthly_budget_twd" in costs.json()
    assert settings.status_code == 200
    assert "sections" in settings.json()


def test_data_refresh_status_endpoints_respond_successfully() -> None:
    client = TestClient(app)

    latest = client.get("/api/data-refresh/latest")
    runs = client.get("/api/data-refresh/runs")

    assert latest.status_code == 200
    assert runs.status_code == 200
    assert isinstance(runs.json(), list)


def test_startup_check_endpoint_responds_successfully() -> None:
    client = TestClient(app)

    response = client.get("/api/system/startup-check")

    assert response.status_code == 200
    payload = response.json()
    assert "overall_status" in payload
    assert "checks" in payload
    assert "providers" in payload
