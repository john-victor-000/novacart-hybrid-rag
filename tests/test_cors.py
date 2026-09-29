from fastapi.testclient import TestClient

from backend.app.main import app


def test_local_frontend_origin_is_allowed() -> None:
    response = TestClient(app).options(
        "/api/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "http://localhost:5173"
    )


def test_unknown_origin_is_not_allowed() -> None:
    response = TestClient(app).options(
        "/api/chat",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert "access-control-allow-origin" not in response.headers
