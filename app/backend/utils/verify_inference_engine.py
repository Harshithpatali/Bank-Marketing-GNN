"""
End-to-end verification of raw-customer GraphSAGE inference.

This script verifies:

raw customer row
    ↓
saved preprocessor
    ↓
temporary HeteroData graph
    ↓
frozen GraphSAGE
    ↓
prediction

No training or fitting occurs.
"""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pathlib import Path

import pandas as pd

from app.backend.services.graph_inference_service import (
    GraphInferenceService,
)
from app.backend.services.inference_engine import (
    InferenceEngine,
)
from app.backend.utils.config import (
    FINAL_MODEL_PATH,
    GRAPH_PATH,
    PREPROCESSOR_PATH,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bank-full.csv"
)


def main() -> None:

    print("=" * 90)
    print("INFERENCE ENGINE VERIFICATION")
    print("=" * 90)

    # --------------------------------------------------------
    # Verify artifacts
    # --------------------------------------------------------

    required = {
        "raw dataset": RAW_DATA_PATH,
        "preprocessor": PREPROCESSOR_PATH,
        "graph": GRAPH_PATH,
        "final model": FINAL_MODEL_PATH,
    }

    for name, path in required.items():

        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found:\n{path}"
            )

        print(
            f"{name:<15}: {path}"
        )

    # --------------------------------------------------------
    # Load one raw customer
    # --------------------------------------------------------

    dataframe = pd.read_csv(
        RAW_DATA_PATH,
        sep=";",
    )

    if dataframe.empty:
        raise ValueError(
            "Raw dataset is empty."
        )

    customer = dataframe.iloc[
        [0]
    ].copy()

    print("\nRaw customer:")
    print(
        customer.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Build inference graph
    # --------------------------------------------------------

    graph_service = GraphInferenceService(
        preprocessor_path=PREPROCESSOR_PATH,
        graph_path=GRAPH_PATH,
    )

    inference_graph, customer_index = (
        graph_service.build_customer_graph(
            customer
        )
    )

    print("\nTemporary graph:")
    print(
        inference_graph
    )

    print(
        "\nNew customer index:",
        customer_index,
    )

    print(
        "Customer feature dimension:",
        inference_graph[
            "customer"
        ].x.shape,
    )

    # --------------------------------------------------------
    # Verify graph structure
    # --------------------------------------------------------

    assert (
        "customer"
        in inference_graph.node_types
    )

    assert (
        inference_graph[
            "customer"
        ].x.shape[1]
        == 50
    )

    assert (
        customer_index
        == 45211
    )

    # --------------------------------------------------------
    # Run frozen inference
    # --------------------------------------------------------

    engine = InferenceEngine(
        model_path=FINAL_MODEL_PATH,
    )

    print(
        "\nModel information:"
    )

    print(
        engine.get_model_info()
    )

    result = engine.predict_graph(
        graph=inference_graph,
        customer_index=customer_index,
    )

    # --------------------------------------------------------
    # Validate prediction
    # --------------------------------------------------------

    probability = result[
        "probability"
    ]

    prediction = result[
        "prediction"
    ]

    assert (
        0.0
        <= probability
        <= 1.0
    )

    assert prediction in (
        0,
        1,
    )

    assert result[
        "prediction_label"
    ] in (
        "yes",
        "no",
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\nPrediction:")
    print(
        result
    )

    print("\n" + "=" * 90)
    print(
        "INFERENCE ENGINE VERIFICATION PASSED"
    )
    print("=" * 90)

    print(
        "Raw customer → preprocessing → graph → "
        "frozen GraphSAGE → prediction: VERIFIED"
    )

    print(
        "Model status:",
        engine.get_model_info()[
            "status"
        ],
    )

    print(
        "Customer feature dimension: 50"
    )

    print(
        "No model training performed."
    )

    print(
        "No preprocessing fitting performed."
    )

    print("=" * 90)


if __name__ == "__main__":
    main()