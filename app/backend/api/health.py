"""
Health and service-status endpoints.
"""

from fastapi import APIRouter

from app.backend.services.prediction_service import (
    PredictionService,
)


router = APIRouter(
    tags=["Health"],
)


# Load the verified frozen prediction service.
prediction_service = PredictionService()


@router.get(
    "/",
)
def root():
    """
    Basic API status.
    """

    model_info = (
        prediction_service.get_model_info()
    )

    return {
        "service": (
            "Bank Marketing GraphSAGE API"
        ),
        "status": "running",
        "model_status": model_info[
            "status"
        ],
    }


@router.get(
    "/health",
)
def health():
    """
    Health check endpoint.
    """

    model_info = (
        prediction_service.get_model_info()
    )

    return {
        "status": "healthy",
        "model_status": model_info[
            "status"
        ],
    }


@router.get(
    "/model-info",
)
def model_info():
    """
    Return information about the frozen model.
    """

    return (
        prediction_service.get_model_info()
    )