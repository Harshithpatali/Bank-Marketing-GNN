"""
Application-level prediction service.

This service coordinates:

Raw customer data
    ↓
GraphInferenceService
    ↓
temporary HeteroData
    ↓
InferenceEngine
    ↓
FINAL_FROZEN_MODEL
    ↓
prediction results

No training or preprocessing fitting occurs here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


class PredictionService:
    """
    High-level prediction service for the frozen GraphSAGE
    model.
    """

    def __init__(
        self,
        model_path: str | Path = FINAL_MODEL_PATH,
        graph_path: str | Path = GRAPH_PATH,
        preprocessor_path: str | Path = PREPROCESSOR_PATH,
    ) -> None:

        self.model_path = Path(
            model_path
        )

        self.graph_path = Path(
            graph_path
        )

        self.preprocessor_path = Path(
            preprocessor_path
        )

        # ----------------------------------------------------
        # Graph construction service
        # ----------------------------------------------------

        self.graph_service = (
            GraphInferenceService(
                preprocessor_path=(
                    self.preprocessor_path
                ),
                graph_path=(
                    self.graph_path
                ),
            )
        )

        # ----------------------------------------------------
        # Frozen model inference engine
        # ----------------------------------------------------

        self.inference_engine = (
            InferenceEngine(
                model_path=self.model_path,
            )
        )

    # ========================================================
    # Input validation
    # ========================================================

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
    ) -> None:

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise ValueError(
                "Input dataframe is empty."
            )

    # ========================================================
    # Single prediction
    # ========================================================

    def predict_single(
        self,
        customer: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Predict one raw customer.

        Parameters
        ----------
        customer:
            Dictionary containing the original Bank Marketing
            customer features.

        Returns
        -------
        dict
            Prediction result.
        """

        if not isinstance(
            customer,
            dict,
        ):
            raise TypeError(
                "customer must be a dictionary."
            )

        # ----------------------------------------------------
        # Build temporary graph
        # ----------------------------------------------------

        graph, customer_index = (
            self.graph_service.prepare_single_customer(
                customer
            )
        )

        # ----------------------------------------------------
        # Frozen GraphSAGE inference
        # ----------------------------------------------------

        result = (
            self.inference_engine.predict_graph(
                graph=graph,
                customer_index=customer_index,
            )
        )

        # ----------------------------------------------------
        # Return application-level response
        # ----------------------------------------------------

        return {
            "probability": result[
                "probability"
            ],
            "prediction": result[
                "prediction"
            ],
            "prediction_label": result[
                "prediction_label"
            ],
            "threshold": result[
                "threshold"
            ],
        }

    # ========================================================
    # Batch prediction
    # ========================================================

    def predict_batch(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict every customer in a raw Bank Marketing
        dataframe.

        The input dataframe is never modified in place.
        """

        self._validate_dataframe(
            dataframe
        )

        working = dataframe.copy()

        # ----------------------------------------------------
        # Preserve original row identifier
        # ----------------------------------------------------

        working.insert(
            0,
            "customer_index",
            range(
                len(working)
            ),
        )

        # ----------------------------------------------------
        # Build inference graphs
        # ----------------------------------------------------

        inference_graphs = (
            self.graph_service.prepare_batch(
                dataframe
            )
        )

        # ----------------------------------------------------
        # Run frozen model
        # ----------------------------------------------------

        predictions = (
            self.inference_engine.predict_graphs(
                inference_graphs
            )
        )

        if len(predictions) != len(
            working
        ):
            raise RuntimeError(
                "Prediction count does not match "
                "input row count."
            )

        # ----------------------------------------------------
        # Construct output
        # ----------------------------------------------------

        output = pd.DataFrame(
            {
                "customer_index": working[
                    "customer_index"
                ].values,

                "prediction_probability": [
                    result[
                        "probability"
                    ]
                    for result in predictions
                ],

                "prediction": [
                    result[
                        "prediction"
                    ]
                    for result in predictions
                ],

                "prediction_label": [
                    result[
                        "prediction_label"
                    ]
                    for result in predictions
                ],
            }
        )

        return output

    # ========================================================
    # Batch prediction with original columns
    # ========================================================

    def predict_batch_with_input(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return the original customer data together with
        prediction results.

        This is useful for downloadable batch prediction CSVs.
        """

        self._validate_dataframe(
            dataframe
        )

        predictions = self.predict_batch(
            dataframe
        )

        original = dataframe.copy()

        original.insert(
            0,
            "customer_index",
            range(
                len(original)
            ),
        )

        return pd.concat(
            [
                original.reset_index(
                    drop=True
                ),
                predictions[
                    [
                        "prediction_probability",
                        "prediction",
                        "prediction_label",
                    ]
                ].reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )

    # ========================================================
    # Model information
    # ========================================================

    def get_model_info(
        self,
    ) -> dict[str, Any]:
        """
        Return information about the frozen model.
        """

        return (
            self.inference_engine.get_model_info()
        )