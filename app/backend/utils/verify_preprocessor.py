"""
Verification utility for the training-fitted preprocessing artifact.

This file performs validation only.
It never fits or modifies the preprocessor.
"""

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "scaler"
    / "preprocessor.joblib"
)

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bank-full.csv"
)


def main() -> None:
    print("=" * 80)
    print("PREPROCESSOR VERIFICATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found:\n{PREPROCESSOR_PATH}"
        )

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_DATA_PATH}"
        )

    print("Preprocessor:", PREPROCESSOR_PATH)
    print("Dataset:", RAW_DATA_PATH)

    # --------------------------------------------------------
    # Load preprocessor
    # --------------------------------------------------------

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    print(
        "Preprocessor type:",
        type(preprocessor).__name__,
    )

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    df = pd.read_csv(
        RAW_DATA_PATH,
        sep=";",
    )

    print(
        "Raw dataset shape:",
        df.shape,
    )

    print(
        "Raw columns:",
        list(df.columns),
    )

    # --------------------------------------------------------
    # Remove target for inference
    # --------------------------------------------------------

    X = df.drop(
        columns=["y"],
        errors="ignore",
    )

    print(
        "Inference feature shape:",
        X.shape,
    )

    # --------------------------------------------------------
    # Transform WITHOUT fitting
    # --------------------------------------------------------

    transformed = preprocessor.transform(X)

    if hasattr(
        transformed,
        "toarray",
    ):
        transformed = transformed.toarray()

    print(
        "Transformed shape:",
        transformed.shape,
    )

    # --------------------------------------------------------
    # Verify expected GraphSAGE dimension
    # --------------------------------------------------------

    expected_dimension = 50

    actual_dimension = transformed.shape[1]

    print(
        "Expected customer feature dimension:",
        expected_dimension,
    )

    print(
        "Actual customer feature dimension:",
        actual_dimension,
    )

    assert (
        actual_dimension
        == expected_dimension
    ), (
        "Preprocessor output dimension does not match "
        "the frozen GraphSAGE customer feature dimension."
    )

    # --------------------------------------------------------
    # Verify numerical validity
    # --------------------------------------------------------

    import numpy as np

    assert np.isfinite(
        transformed
    ).all(), (
        "Preprocessor produced NaN or infinite values."
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("=" * 80)
    print("PREPROCESSOR VERIFICATION PASSED")
    print("=" * 80)
    print(
        "Rows transformed:",
        transformed.shape[0],
    )
    print(
        "Features produced:",
        transformed.shape[1],
    )
    print(
        "Fitted preprocessor reused:",
        True,
    )
    print(
        "Preprocessor refitted:",
        False,
    )
    print("=" * 80)


if __name__ == "__main__":
    main()