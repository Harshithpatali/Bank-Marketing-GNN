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

    The original Bank Marketing columns are accepted.

    The target column 'y' is optional and ignored during
    inference.

    Returns a downloadable prediction CSV.
    """

    # --------------------------------------------------------
    # Validate filename
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
    # Read uploaded CSV
    # --------------------------------------------------------

    try:

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV is empty.",
            )

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
    # Validate data
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
    # Remove target if present
    # --------------------------------------------------------

    inference_dataframe = dataframe.drop(
        columns=["y"],
        errors="ignore",
    )

    # --------------------------------------------------------
    # Run frozen GraphSAGE inference
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
    # Convert predictions to CSV
    # --------------------------------------------------------

    output_buffer = io.StringIO()

    results.to_csv(
        output_buffer,
        index=False,
    )

    output_buffer.seek(0)

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
                "filename="
                "bank_marketing_predictions.csv"
            )
        },
    )