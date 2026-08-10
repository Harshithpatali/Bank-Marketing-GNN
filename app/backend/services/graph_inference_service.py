"""
Production graph inference service.

Converts raw Bank Marketing customer records into the exact
customer-node representation expected by the frozen
heterogeneous GraphSAGE model.

Important:
- Reuses the training-fitted preprocessor.
- Never fits preprocessing during inference.
- Uses verified graph category mappings.
- Never modifies the saved graph artifact.
- Never trains or modifies the frozen model.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from app.backend.utils.config import (
    GRAPH_PATH,
    PREPROCESSOR_PATH,
)


class GraphInferenceService:
    """
    Production preprocessing and graph construction service.

    Raw customer data
        ↓
    saved preprocessor
        ↓
    50-dimensional customer representation
        ↓
    temporary customer node
        ↓
    verified categorical graph connections
        ↓
    GraphSAGE-ready HeteroData
    """

    # ========================================================
    # Verified mappings recovered from the saved graph.
    # ========================================================

    CATEGORY_MAPPINGS = {
        "job": {
            "admin.": 0,
            "blue-collar": 1,
            "entrepreneur": 2,
            "housemaid": 3,
            "management": 4,
            "retired": 5,
            "self-employed": 6,
            "services": 7,
            "student": 8,
            "technician": 9,
            "unemployed": 10,
            "unknown": 11,
        },
        "education": {
            "primary": 0,
            "secondary": 1,
            "tertiary": 2,
            "unknown": 3,
        },
        "marital": {
            "divorced": 0,
            "married": 1,
            "single": 2,
        },
        "contact": {
            "cellular": 0,
            "telephone": 1,
            "unknown": 2,
        },
        "month": {
            "apr": 0,
            "aug": 1,
            "dec": 2,
            "feb": 3,
            "jan": 4,
            "jul": 5,
            "jun": 6,
            "mar": 7,
            "may": 8,
            "nov": 9,
            "oct": 10,
            "sep": 11,
        },
    }

    RELATIONS = {
        "job": (
            "customer",
            "has_job",
            "job",
            "rev_has_job",
        ),
        "education": (
            "customer",
            "has_education",
            "education",
            "rev_has_education",
        ),
        "marital": (
            "customer",
            "has_marital_status",
            "marital",
            "rev_has_marital_status",
        ),
        "contact": (
            "customer",
            "contacted_via",
            "contact",
            "rev_contacted_via",
        ),
        "month": (
            "customer",
            "campaign_month",
            "month",
            "rev_campaign_month",
        ),
    }

    REQUIRED_RAW_COLUMNS = {
        "job",
        "education",
        "marital",
        "contact",
        "month",
    }

    def __init__(
        self,
        preprocessor_path: str | Path = PREPROCESSOR_PATH,
        graph_path: str | Path = GRAPH_PATH,
    ) -> None:

        self.preprocessor_path = Path(
            preprocessor_path
        )

        self.graph_path = Path(
            graph_path
        )

        self.preprocessor: Any = None
        self.base_graph: HeteroData | None = None

        self._load_artifacts()
        self._validate_mappings()

    # ========================================================
    # Load artifacts
    # ========================================================

    def _load_artifacts(self) -> None:
        """Load the saved preprocessor and graph."""

        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                "Preprocessor not found: "
                f"{self.preprocessor_path}"
            )

        if not self.graph_path.exists():
            raise FileNotFoundError(
                "Graph not found: "
                f"{self.graph_path}"
            )

        self.preprocessor = joblib.load(
            self.preprocessor_path
        )

        try:
            self.base_graph = torch.load(
                self.graph_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            self.base_graph = torch.load(
                self.graph_path,
                map_location="cpu",
            )

        if not isinstance(
            self.base_graph,
            HeteroData,
        ):
            raise TypeError(
                "Graph artifact must be HeteroData."
            )

    # ========================================================
    # Validate mappings against graph
    # ========================================================

    def _validate_mappings(self) -> None:
        """
        Verify that the hard-coded verified mappings still match
        the node counts of the saved graph.
        """

        if self.base_graph is None:
            raise RuntimeError(
                "Base graph has not been loaded."
            )

        for category, mapping in (
            self.CATEGORY_MAPPINGS.items()
        ):

            if category not in (
                self.base_graph.node_types
            ):
                raise ValueError(
                    f"Graph missing node type: {category}"
                )

            graph_count = (
                self.base_graph[
                    category
                ].num_nodes
            )

            mapping_count = len(mapping)

            if graph_count != mapping_count:
                raise ValueError(
                    f"Mapping mismatch for {category}: "
                    f"graph has {graph_count} nodes, "
                    f"mapping has {mapping_count} values."
                )

            indices = sorted(
                mapping.values()
            )

            expected = list(
                range(graph_count)
            )

            if indices != expected:
                raise ValueError(
                    f"Mapping indices for {category} "
                    "are not contiguous."
                )

    # ========================================================
    # Raw input validation
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
                "Input dataframe contains no rows."
            )

    def validate_batch_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate columns required for raw graph inference."""

        self._validate_dataframe(
            dataframe
        )

        missing = (
            self.REQUIRED_RAW_COLUMNS
            - set(dataframe.columns)
        )

        if missing:
            raise ValueError(
                "Raw input is missing required "
                f"columns: {sorted(missing)}"
            )

    # ========================================================
    # Preprocessing
    # ========================================================

    def transform_customer_features(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """
        Transform raw customer records with the fitted
        training preprocessor.

        No fitting occurs here.
        """

        self._validate_dataframe(
            dataframe
        )

        working = dataframe.copy()

        if "y" in working.columns:
            working = working.drop(
                columns=["y"]
            )

        try:
            transformed = (
                self.preprocessor.transform(
                    working
                )
            )
        except Exception as exc:
            raise ValueError(
                "Input could not be transformed by "
                "the saved training preprocessor."
            ) from exc

        if hasattr(
            transformed,
            "toarray",
        ):
            transformed = transformed.toarray()

        transformed = np.asarray(
            transformed,
            dtype=np.float32,
        )

        if transformed.ndim != 2:
            raise ValueError(
                "Preprocessor output must be 2-dimensional."
            )

        if transformed.shape[1] != 50:
            raise ValueError(
                "Preprocessor output dimension mismatch: "
                f"expected 50, got {transformed.shape[1]}."
            )

        if not np.isfinite(
            transformed
        ).all():
            raise ValueError(
                "Preprocessor produced NaN or infinite values."
            )

        return transformed

    # ========================================================
    # Category lookup
    # ========================================================

    def _category_index(
        self,
        category: str,
        value: Any,
    ) -> int:
        """
        Convert a raw categorical value to its verified graph
        node index.
        """

        normalized = str(value)

        mapping = self.CATEGORY_MAPPINGS[
            category
        ]

        if normalized not in mapping:
            allowed = sorted(
                mapping.keys()
            )

            raise ValueError(
                f"Unknown {category} value: "
                f"{normalized!r}. "
                f"Allowed values: {allowed}"
            )

        return mapping[
            normalized
        ]

    # ========================================================
    # Append edge
    # ========================================================

    @staticmethod
    def _append_edge(
        edge_index: torch.Tensor,
        source_index: int,
        target_index: int,
    ) -> torch.Tensor:

        new_edge = torch.tensor(
            [
                [source_index],
                [target_index],
            ],
            dtype=edge_index.dtype,
        )

        return torch.cat(
            [
                edge_index.cpu(),
                new_edge,
            ],
            dim=1,
        )

    # ========================================================
    # Build one customer graph
    # ========================================================

    def build_customer_graph(
        self,
        raw_customer: pd.DataFrame,
    ) -> tuple[HeteroData, int]:
        """
        Build a temporary graph containing one new customer.

        The saved base graph itself is never modified.
        """

        if len(raw_customer) != 1:
            raise ValueError(
                "Exactly one customer row is required."
            )

        self.validate_batch_columns(
            raw_customer
        )

        transformed = (
            self.transform_customer_features(
                raw_customer
            )
        )

        if self.base_graph is None:
            raise RuntimeError(
                "Base graph has not been loaded."
            )

        graph = deepcopy(
            self.base_graph.cpu()
        )

        customer_index = (
            graph["customer"].num_nodes
        )

        customer_feature = torch.tensor(
            transformed[0],
            dtype=torch.float32,
        ).unsqueeze(0)

        expected_features = (
            graph["customer"].x.size(1)
        )

        if customer_feature.size(1) != (
            expected_features
        ):
            raise ValueError(
                "Customer feature dimension mismatch: "
                f"expected {expected_features}, "
                f"got {customer_feature.size(1)}."
            )

        graph["customer"].x = torch.cat(
            [
                graph["customer"].x.cpu(),
                customer_feature,
            ],
            dim=0,
        )

        # ----------------------------------------------------
        # Connect new customer to categorical nodes.
        # ----------------------------------------------------

        for category, relations in (
            self.RELATIONS.items()
        ):

            (
                source_type,
                forward_relation,
                target_type,
                reverse_relation,
            ) = relations

            raw_value = raw_customer.iloc[0][
                category
            ]

            target_index = (
                self._category_index(
                    category,
                    raw_value,
                )
            )

            forward_edge = (
                source_type,
                forward_relation,
                target_type,
            )

            reverse_edge = (
                target_type,
                reverse_relation,
                source_type,
            )

            if forward_edge not in (
                graph.edge_types
            ):
                raise ValueError(
                    f"Missing graph relation: "
                    f"{forward_edge}"
                )

            if reverse_edge not in (
                graph.edge_types
            ):
                raise ValueError(
                    f"Missing graph relation: "
                    f"{reverse_edge}"
                )

            graph[
                forward_edge
            ].edge_index = self._append_edge(
                graph[
                    forward_edge
                ].edge_index,
                customer_index,
                target_index,
            )

            graph[
                reverse_edge
            ].edge_index = self._append_edge(
                graph[
                    reverse_edge
                ].edge_index,
                target_index,
                customer_index,
            )

        return (
            graph,
            customer_index,
        )

    # ========================================================
    # Prepare one raw customer
    # ========================================================

    def prepare_single_customer(
        self,
        customer: dict[str, Any],
    ) -> tuple[HeteroData, int]:

        dataframe = pd.DataFrame(
            [customer]
        )

        return self.build_customer_graph(
            dataframe
        )

    # ========================================================
    # Prepare batch
    # ========================================================

    def prepare_batch(
        self,
        dataframe: pd.DataFrame,
    ) -> list[tuple[HeteroData, int]]:
        """
        Convert raw customer rows into individual inference
        graphs.
        """

        self.validate_batch_columns(
            dataframe
        )

        prepared = []

        for _, row in dataframe.iterrows():

            customer_df = pd.DataFrame(
                [row.to_dict()]
            )

            prepared.append(
                self.build_customer_graph(
                    customer_df
                )
            )

        return prepared