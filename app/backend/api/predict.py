"""
Prediction API routes for the Bank Marketing GraphSAGE backend.
"""

from fastapi import APIRouter, HTTPException

from app.backend.schemas.prediction import (
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.backend.services.prediction_service import (
    PredictionService,
)
from app.backend.utils.config import (
    FINAL_MODEL_PATH,
    GRAPH_PATH,
)


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


# ------------------------------------------------------------
# Load the frozen inference service once.
# ------------------------------------------------------------

prediction_service = PredictionService(
    model_path=FINAL_MODEL_PATH,
    graph_path=GRAPH_PATH,
)


@router.post(
    "",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    """
    Generate a prediction for an existing customer node.
    """

    try:
        result = prediction_service.predict_customer(
            customer_index=request.customer_index
        )

        return PredictionResponse(
            **result
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info() -> ModelInfoResponse:
    """
    Return information about the deployed frozen model.
    """

    try:
        result = prediction_service.get_model_info()

        return ModelInfoResponse(
            **result
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load model information: {exc}",
        ) from exc