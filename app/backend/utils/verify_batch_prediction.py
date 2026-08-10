"""
Verify batch prediction using five real Bank Marketing rows.

This test:
- uses the saved preprocessor
- uses verified graph mappings
- uses the frozen GraphSAGE model
- does not train or fit anything
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
    print("BATCH PREDICTION VERIFICATION")
    print("=" * 90)

    # --------------------------------------------------------
    # Load real dataset
    # --------------------------------------------------------

    dataframe = pd.read_csv(
        RAW_DATA_PATH,
        sep=";",
    )

    batch = dataframe.iloc[
        :5
    ].copy()

    print(
        "Input rows:",
        len(batch),
    )

    # --------------------------------------------------------
    # Prediction service
    # --------------------------------------------------------

    service = PredictionService()

    # --------------------------------------------------------
    # Run batch prediction
    # --------------------------------------------------------

    results = service.predict_batch_with_input(
        batch
    )

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    print("\nPrediction output:")
    print(
        results.to_string(
            index=False
        )
    )

    assert len(results) == 5

    required_columns = {
        "customer_index",
        "prediction_probability",
        "prediction",
        "prediction_label",
    }

    missing = (
        required_columns
        - set(results.columns)
    )

    assert not missing, (
        f"Missing output columns: {missing}"
    )

    assert results[
        "prediction_probability"
    ].between(
        0.0,
        1.0,
    ).all()

    assert results[
        "prediction"
    ].isin(
        [0, 1]
    ).all()

    assert results[
        "prediction_label"
    ].isin(
        ["yes", "no"]
    ).all()

    assert results[
        "customer_index"
    ].tolist() == [
        0,
        1,
        2,
        3,
        4,
    ]

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    print("\nModel information:")
    print(
        service.get_model_info()
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print(
        "BATCH PREDICTION VERIFICATION PASSED"
    )
    print("=" * 90)

    print(
        "Rows processed:",
        len(results),
    )

    print(
        "Predictions generated:",
        len(results),
    )

    print(
        "Output schema: VERIFIED"
    )

    print(
        "Model status:",
        service.get_model_info()[
            "status"
        ],
    )

    print("=" * 90)


if __name__ == "__main__":
    main()