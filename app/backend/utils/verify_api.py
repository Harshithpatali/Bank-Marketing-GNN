from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def main():

    print("=" * 90)
    print("FASTAPI APPLICATION VERIFICATION")
    print("=" * 90)

    # --------------------------------------------------------
    # Root
    # --------------------------------------------------------

    response = client.get("/")

    print("\nGET /")
    print("Status:", response.status_code)
    print("Response:", response.json())

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"
    assert data["model_status"] == "FINAL_FROZEN_MODEL"

    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    response = client.get("/health")

    print("\nGET /health")
    print("Status:", response.status_code)
    print("Response:", response.json())

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_status"] == "FINAL_FROZEN_MODEL"

    # --------------------------------------------------------
    # Route registration
    # --------------------------------------------------------

    routes = {
        route.path
        for route in app.routes
    }

    print("\nRegistered routes:")

    for route in sorted(routes):
        print(" ", route)

    assert "/batch/predict" in routes

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("FASTAPI APPLICATION VERIFICATION PASSED")
    print("=" * 90)

    print("Root endpoint: VERIFIED")
    print("Health endpoint: VERIFIED")
    print("Batch prediction route: VERIFIED")
    print("=" * 90)


if __name__ == "__main__":
    main()