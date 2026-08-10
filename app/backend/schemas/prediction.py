"""
Prediction request and response schemas.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request for prediction on an existing customer node.
    """

    customer_index: int = Field(
        ...,
        ge=0,
        description="Customer node index in the saved heterogeneous graph.",
    )


class PredictionResponse(BaseModel):
    """
    Prediction returned by the frozen GraphSAGE model.
    """

    customer_index: int
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )
    prediction: int = Field(
        ...,
        ge=0,
        le=1,
    )
    prediction_label: str
    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )


class ModelInfoResponse(BaseModel):
    """
    Metadata about the deployed frozen model.
    """

    status: str
    hidden_dim: int
    dropout: float
    learning_rate: float
    classification_threshold: float
    device: str