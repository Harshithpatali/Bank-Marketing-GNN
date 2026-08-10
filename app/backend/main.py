"""
FastAPI application entry point for the Bank Marketing GraphSAGE backend.
"""

from fastapi import FastAPI

from app.backend.api.batch import router as batch_router
from app.backend.api.health import router as health_router
from app.backend.api.predict import router as prediction_router


app = FastAPI(
    title="Bank Marketing GraphSAGE API",
    description=(
        "REST API for bank marketing subscription prediction "
        "using a frozen heterogeneous GraphSAGE model."
    ),
    version="1.0.0",
)


# ------------------------------------------------------------
# API routers
# ------------------------------------------------------------

app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(batch_router)