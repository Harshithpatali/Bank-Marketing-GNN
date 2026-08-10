"""
Inference engine for the frozen heterogeneous GraphSAGE model.

This module:
- loads the frozen checkpoint
- reconstructs the exact HeteroGraphSAGE architecture
- accepts arbitrary HeteroData inference graphs
- performs inference only

No training, fitting, optimization, or checkpoint modification occurs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGraphSAGE(nn.Module):
    """
    Frozen GraphSAGE architecture used by Notebook 05C/06.

    Architecture:

        HeteroConv
            ↓
        ReLU
            ↓
        Dropout
            ↓
        HeteroConv
            ↓
        ReLU
            ↓
        Dropout
            ↓
        Linear classifier
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_dim: int = 128,
        dropout: float = 0.20,
    ) -> None:

        super().__init__()

        _, edge_types = metadata

        self.dropout = dropout

        # ----------------------------------------------------
        # First GraphSAGE layer
        # ----------------------------------------------------

        self.conv1 = HeteroConv(
            {
                edge_type: SAGEConv(
                    (-1, -1),
                    hidden_dim,
                    aggr="mean",
                )
                for edge_type in edge_types
            },
            aggr="sum",
        )

        # ----------------------------------------------------
        # Second GraphSAGE layer
        # ----------------------------------------------------

        self.conv2 = HeteroConv(
            {
                edge_type: SAGEConv(
                    (-1, -1),
                    hidden_dim,
                    aggr="mean",
                )
                for edge_type in edge_types
            },
            aggr="sum",
        )

        # ----------------------------------------------------
        # Customer classifier
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[
            tuple[str, str, str],
            torch.Tensor,
        ],
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # GraphSAGE layer 1
        # ----------------------------------------------------

        x_dict = self.conv1(
            x_dict,
            edge_index_dict,
        )

        x_dict = {
            node_type: F.dropout(
                F.relu(features),
                p=self.dropout,
                training=self.training,
            )
            for node_type, features
            in x_dict.items()
        }

        # ----------------------------------------------------
        # GraphSAGE layer 2
        # ----------------------------------------------------

        x_dict = self.conv2(
            x_dict,
            edge_index_dict,
        )

        x_dict = {
            node_type: F.dropout(
                F.relu(features),
                p=self.dropout,
                training=self.training,
            )
            for node_type, features
            in x_dict.items()
        }

        # ----------------------------------------------------
        # Customer classification
        # ----------------------------------------------------

        return self.classifier(
            x_dict["customer"]
        ).squeeze(-1)


class InferenceEngine:
    """
    Loads the FINAL_FROZEN_MODEL checkpoint and performs
    inference on supplied HeteroData graphs.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: torch.device | None = None,
    ) -> None:

        self.model_path = Path(
            model_path
        )

        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.checkpoint: dict[str, Any] = {}

        self.model_config: dict[str, Any] = {}

        self._load_checkpoint()

    # ========================================================
    # Checkpoint loading
    # ========================================================

    def _load_checkpoint(self) -> None:
        """
        Load the frozen checkpoint.

        The model itself is constructed only after an actual
        HeteroData graph is supplied because the graph metadata
        defines the exact typed edge structure.
        """

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Final model not found: "
                f"{self.model_path}"
            )

        try:
            checkpoint = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                self.model_path,
                map_location="cpu",
            )

        if not isinstance(
            checkpoint,
            dict,
        ):
            raise TypeError(
                "Frozen model checkpoint must be a dictionary."
            )

        self.checkpoint = checkpoint

        # ----------------------------------------------------
        # Freeze contract
        # ----------------------------------------------------

        freeze_status = checkpoint.get(
            "freeze_status"
        )

        if freeze_status != (
            "FINAL_FROZEN_MODEL"
        ):
            raise ValueError(
                "InferenceEngine requires "
                "FINAL_FROZEN_MODEL. "
                f"Received: {freeze_status!r}"
            )

        # ----------------------------------------------------
        # Required checkpoint fields
        # ----------------------------------------------------

        required_keys = {
            "model_state_dict",
            "model_config",
            "graph_metadata",
            "classification_threshold",
        }

        missing = (
            required_keys
            - checkpoint.keys()
        )

        if missing:
            raise ValueError(
                "Frozen checkpoint is missing required "
                f"fields: {sorted(missing)}"
            )

        self.model_config = checkpoint[
            "model_config"
        ]

        if (
            self.model_config.get(
                "model_class"
            )
            != "HeteroGraphSAGE"
        ):
            raise ValueError(
                "Unexpected model class: "
                f"{self.model_config.get('model_class')}"
            )

    # ========================================================
    # Graph validation
    # ========================================================

    def _validate_graph(
        self,
        graph: HeteroData,
    ) -> None:

        if not isinstance(
            graph,
            HeteroData,
        ):
            raise TypeError(
                "Inference graph must be HeteroData."
            )

        if "customer" not in (
            graph.node_types
        ):
            raise ValueError(
                "Inference graph is missing "
                "the customer node type."
            )

        customer_x = graph[
            "customer"
        ].x

        if customer_x is None:
            raise ValueError(
                "Customer node features are missing."
            )

        expected_dimension = self.model_config[
            "customer_feature_dimension"
        ]

        actual_dimension = customer_x.size(
            1
        )

        if actual_dimension != (
            expected_dimension
        ):
            raise ValueError(
                "Customer feature dimension mismatch: "
                f"expected {expected_dimension}, "
                f"got {actual_dimension}."
            )

        # ----------------------------------------------------
        # Verify graph metadata against checkpoint
        # ----------------------------------------------------

        checkpoint_metadata = (
            self.checkpoint[
                "model_config"
            ]
        )

        expected_node_types = set(
            checkpoint_metadata[
                "node_types"
            ]
        )

        actual_node_types = set(
            graph.node_types
        )

        if actual_node_types != (
            expected_node_types
        ):
            raise ValueError(
                "Node-type mismatch between inference graph "
                "and frozen model."
            )

        expected_edge_types = {
            tuple(edge)
            for edge in checkpoint_metadata[
                "edge_types"
            ]
        }

        actual_edge_types = set(
            graph.edge_types
        )

        if actual_edge_types != (
            expected_edge_types
        ):
            raise ValueError(
                "Edge-type mismatch between inference graph "
                "and frozen model."
            )

    # ========================================================
    # Construct exact frozen architecture
    # ========================================================

    def _build_frozen_model(
        self,
        graph: HeteroData,
    ) -> HeteroGraphSAGE:
        """
        Construct the exact architecture using the metadata
        of the supplied inference graph, initialize lazy
        parameters, then load the frozen state dictionary.
        """

        self._validate_graph(
            graph
        )

        hidden_dim = self.model_config[
            "hidden_dim"
        ]

        dropout = self.model_config[
            "dropout"
        ]

        model = HeteroGraphSAGE(
            metadata=graph.metadata(),
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(
            self.device
        )

        # ----------------------------------------------------
        # Initialize lazy SAGEConv parameters.
        # ----------------------------------------------------

        graph_device = graph.to(
            self.device
        )

        model.eval()

        with torch.no_grad():
            _ = model(
                graph_device.x_dict,
                graph_device.edge_index_dict,
            )

        # ----------------------------------------------------
        # Load exact frozen parameters.
        # ----------------------------------------------------

        state_dict = self.checkpoint[
            "model_state_dict"
        ]

        try:
            model.load_state_dict(
                state_dict,
                strict=True,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Frozen GraphSAGE state_dict does not match "
                "the reconstructed architecture. "
                "The backend architecture must match the "
                "training architecture exactly."
            ) from exc

        model.eval()

        return model

    # ========================================================
    # Single graph prediction
    # ========================================================

    def predict_graph(
        self,
        graph: HeteroData,
        customer_index: int,
    ) -> dict[str, Any]:
        """
        Run frozen GraphSAGE inference on one graph.
        """

        self._validate_graph(
            graph
        )

        customer_count = (
            graph[
                "customer"
            ].num_nodes
        )

        if not (
            0
            <= customer_index
            < customer_count
        ):
            raise ValueError(
                "customer_index is outside "
                "the inference graph."
            )

        graph = graph.to(
            self.device
        )

        model = self._build_frozen_model(
            graph
        )

        with torch.no_grad():

            logits = model(
                graph.x_dict,
                graph.edge_index_dict,
            )

            probabilities = torch.sigmoid(
                logits
            )

        threshold = float(
            self.checkpoint[
                "classification_threshold"
            ]
        )

        probability = float(
            probabilities[
                customer_index
            ].item()
        )

        prediction = int(
            probability >= threshold
        )

        return {
            "customer_index": int(
                customer_index
            ),
            "probability": probability,
            "prediction": prediction,
            "prediction_label": (
                "yes"
                if prediction == 1
                else "no"
            ),
            "threshold": threshold,
        }

    # ========================================================
    # Batch graph prediction
    # ========================================================

    def predict_graphs(
        self,
        graphs: list[
            tuple[HeteroData, int]
        ],
    ) -> list[dict[str, Any]]:

        results = []

        for graph, customer_index in graphs:

            results.append(
                self.predict_graph(
                    graph=graph,
                    customer_index=customer_index,
                )
            )

        return results

    # ========================================================
    # Model information
    # ========================================================

    def get_model_info(
        self,
    ) -> dict[str, Any]:

        return {
            "status": self.checkpoint[
                "freeze_status"
            ],
            "hidden_dim": self.model_config[
                "hidden_dim"
            ],
            "dropout": self.model_config[
                "dropout"
            ],
            "learning_rate": self.model_config[
                "learning_rate"
            ],
            "classification_threshold": self.checkpoint[
                "classification_threshold"
            ],
            "device": str(
                self.device
            ),
        }