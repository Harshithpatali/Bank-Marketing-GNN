"""
Verify the application-level PredictionService.

Runs one real customer through the complete prediction stack.
"""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.backend.services.prediction_service import (
    PredictionService,
)


RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bank-full.csv"
)


def main():

    print("=" * 90)
    print("PREDICTION SERVICE VERIFICATION")
    print("=" * 90)

    dataframe = pd.read_csv(
        RAW_DATA_PATH,
        sep=";",
    )

    customer = dataframe.iloc[
        0
    ].drop(
        labels=["y"],
        errors="ignore",
    ).to_dict()

    service = PredictionService()

    result = service.predict_single(
        customer
    )

    print("\nPrediction result:")
    print(result)

    assert (
        0.0
        <= result[
            "probability"
        ]
        <= 1.0
    )

    assert result[
        "prediction"
    ] in (0, 1)

    assert result[
        "prediction_label"
    ] in ("yes", "no")

    print("\nModel information:")
    print(
        service.get_model_info()
    )

    print("\n" + "=" * 90)
    print(
        "PREDICTION SERVICE VERIFICATION PASSED"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()