"""
Verify the FastAPI /batch/predict endpoint with 5 real rows.
"""

from pathlib import Path
import io
import sys

import pandas as pd
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.backend.main import app


RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bank-full.csv"
)


def main():

    print("=" * 90)
    print("FASTAPI BATCH PREDICTION VERIFICATION")
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
    # Create temporary CSV in memory
    # --------------------------------------------------------

    csv_buffer = io.StringIO()

    batch.to_csv(
        csv_buffer,
        index=False,
        sep=";",
    )

    csv_bytes = (
        csv_buffer.getvalue()
        .encode("utf-8")
    )

    # --------------------------------------------------------
    # Test FastAPI endpoint
    # --------------------------------------------------------

    client = TestClient(
        app
    )

    response = client.post(
        "/batch/predict",
        files={
            "file": (
                "bank-full-test.csv",
                csv_bytes,
                "text/csv",
            )
        },
    )

    print(
        "\nHTTP status:",
        response.status_code,
    )

    print(
        "Content-Type:",
        response.headers.get(
            "content-type"
        ),
    )

    # --------------------------------------------------------
    # Validate HTTP response
    # --------------------------------------------------------

    assert response.status_code == 200, (
        "Batch endpoint failed:\n"
        f"{response.text}"
    )

    content_type = response.headers.get(
        "content-type",
        ""
    )

    assert "text/csv" in content_type

    # --------------------------------------------------------
    # Read returned CSV
    # --------------------------------------------------------

    result = pd.read_csv(
        io.BytesIO(
            response.content
        )
    )

    print(
        "\nReturned columns:"
    )

    for column in result.columns:
        print(
            " ",
            column,
        )

    print(
        "\nReturned predictions:"
    )

    print(
        result.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Validate result
    # --------------------------------------------------------

    assert len(result) == 5

    required_columns = {
        "customer_index",
        "prediction_probability",
        "prediction",
        "prediction_label",
    }

    missing = (
        required_columns
        - set(result.columns)
    )

    assert not missing, (
        f"Missing output columns: {missing}"
    )

    assert result[
        "customer_index"
    ].tolist() == [
        0,
        1,
        2,
        3,
        4,
    ]

    assert result[
        "prediction_probability"
    ].between(
        0.0,
        1.0,
    ).all()

    assert result[
        "prediction"
    ].isin(
        [0, 1]
    ).all()

    assert result[
        "prediction_label"
    ].isin(
        ["yes", "no"]
    ).all()

    # --------------------------------------------------------
    # Validate download header
    # --------------------------------------------------------

    disposition = response.headers.get(
        "content-disposition",
        ""
    )

    print(
        "\nContent-Disposition:",
        disposition,
    )

    assert (
        "bank_marketing_predictions.csv"
        in disposition
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print(
        "FASTAPI BATCH PREDICTION VERIFICATION PASSED"
    )
    print("=" * 90)

    print(
        "HTTP status: 200"
    )

    print(
        "Rows returned:",
        len(result),
    )

    print(
        "CSV response: VERIFIED"
    )

    print(
        "Download filename: VERIFIED"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()