from __future__ import annotations

from fastapi.testclient import TestClient

from tw_stock_ai.main import app


def test_ui_pages_respond_successfully() -> None:
    client = TestClient(app)

    for path in ("/picks", "/treasures", "/holdings", "/settings", "/system"):
        response = client.get(path)
        assert response.status_code == 200


def test_ui_pages_render_expected_labels_without_broken_tags() -> None:
    client = TestClient(app)

    expected_labels = {
        "/picks": "今日選股",
        "/treasures": "寶藏股",
        "/holdings": "我的持股",
        "/settings": "設定",
        "/system": "系統狀態",
    }

    for path, label in expected_labels.items():
        response = client.get(path)
        assert response.status_code == 200
        assert label in response.text
        assert "?/p>" not in response.text
        assert "?/a>" not in response.text


def test_picks_page_shows_recent_news_panel() -> None:
    client = TestClient(app)

    response = client.get("/picks")

    assert response.status_code == 200
    assert "Recent Market News" in response.text
