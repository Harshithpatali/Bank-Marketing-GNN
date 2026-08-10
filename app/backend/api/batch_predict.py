"""
FastAPI batch prediction endpoint.

Accepts the original Bank Marketing CSV and returns
GraphSAGE predictions as a downloadable CSV.

Endpoint:
    POST /batch/predict
"""

from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.backend.services.prediction_service import (
    PredictionService,
)


router = APIRouter(
    prefix="/batch",
    tags=["Batch Prediction"],
)


# ------------------------------------------------------------
# Required raw Bank Marketing columns
# ------------------------------------------------------------

REQUIRED_COLUMNS = {
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
}


# ------------------------------------------------------------
# Prediction service
# ------------------------------------------------------------

prediction_service = PredictionService()


@router.post(
    "/predict",
    summary="Batch predict from Bank Marketing CSV",
)
async def batch_predict(
    file: UploadFile = File(...),
):
    """
    Run batch GraphSAGE predictions on a Bank Marketing CSV.

    The uploaded CSV must contain the original Bank Marketing
    feature columns.

    The target column 'y' is optional and ignored if present.

    Returns:
        Downloadable CSV containing original customer data
        plus prediction columns.
    """

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename supplied.",
        )

    if not file.filename.lower().endswith(
        ".csv"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    try:

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV is empty.",
            )

        # Bank Marketing dataset uses ';'
        # but accepting comma-separated CSVs
        # makes the endpoint more robust.

        sample = contents[:5000].decode(
            "utf-8",
            errors="replace",
        )

        separator = (
            ";"
            if sample.count(";")
            > sample.count(",")
            else ","
        )

        dataframe = pd.read_csv(
            io.BytesIO(contents),
            sep=separator,
        )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read uploaded CSV: "
                f"{exc}"
            ),
        ) from exc

    # --------------------------------------------------------
    # Validate dataframe
    # --------------------------------------------------------

    if dataframe.empty:
        raise HTTPException(
            status_code=400,
            detail="CSV contains no data rows.",
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            ),
        )

    # --------------------------------------------------------
    # Remove target from inference
    # --------------------------------------------------------

    inference_dataframe = dataframe.drop(
        columns=["y"],
        errors="ignore",
    )

    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    try:

        results = (
            prediction_service
            .predict_batch_with_input(
                inference_dataframe
            )
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Batch prediction failed: "
                f"{exc}"
            ),
        ) from exc

    # --------------------------------------------------------
    # Convert results to CSV
    # --------------------------------------------------------

    output_buffer = io.StringIO()

    results.to_csv(
        output_buffer,
        index=False,
    )

    output_buffer.seek(0)

    output_filename = (
        "bank_marketing_predictions.csv"
    )

    # --------------------------------------------------------
    # Return downloadable CSV
    # --------------------------------------------------------

    return StreamingResponse(
        iter(
            [
                output_buffer.getvalue()
            ]
        ),
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                "attachment; "
                f"filename={output_filename}"
            )
        },
    )