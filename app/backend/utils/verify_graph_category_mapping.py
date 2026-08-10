"""
Recover and VERIFY categorical graph-node mappings from the
saved graph and the original Bank Marketing dataset.

This script does NOT modify the graph.

It uses the customer -> categorical-node edges already stored
in HeteroData and verifies that every node corresponds to one
raw categorical value.
"""

from pathlib import Path

from collections import defaultdict

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]

GRAPH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "graph"
    / "bank_heterodata.pt"
)

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bank-full.csv"
)


RELATIONS = {
    "job": (
        "customer",
        "has_job",
        "job",
    ),
    "education": (
        "customer",
        "has_education",
        "education",
    ),
    "marital": (
        "customer",
        "has_marital_status",
        "marital",
    ),
    "contact": (
        "customer",
        "contacted_via",
        "contact",
    ),
    "month": (
        "customer",
        "campaign_month",
        "month",
    ),
}


def load_graph():

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Graph not found:\n{GRAPH_PATH}"
        )

    try:
        return torch.load(
            GRAPH_PATH,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            GRAPH_PATH,
            map_location="cpu",
        )


def main():

    print("=" * 90)
    print("GRAPH CATEGORY MAPPING RECOVERY + VERIFICATION")
    print("=" * 90)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    graph = load_graph()

    dataframe = pd.read_csv(
        RAW_DATA_PATH,
        sep=";",
    )

    print(
        "Raw dataset rows:",
        len(dataframe),
    )

    print(
        "Graph customer nodes:",
        graph["customer"].num_nodes,
    )

    assert len(dataframe) == (
        graph["customer"].num_nodes
    ), (
        "Raw dataset row count does not match "
        "customer-node count."
    )

    # --------------------------------------------------------
    # Recover mapping from graph edges
    # --------------------------------------------------------

    all_mappings = {}

    for category, edge_type in RELATIONS.items():

        print("\n" + "=" * 90)
        print(f"CATEGORY: {category}")
        print("=" * 90)

        if edge_type not in graph.edge_types:
            raise ValueError(
                f"Missing graph relation: {edge_type}"
            )

        edge_index = (
            graph[edge_type]
            .edge_index
            .cpu()
            .numpy()
        )

        customer_indices = edge_index[0]
        category_indices = edge_index[1]

        assert len(customer_indices) == len(
            dataframe
        ), (
            f"{category}: edge count does not match "
            "number of customers."
        )

        mapping_values = defaultdict(set)

        for customer_index, node_index in zip(
            customer_indices,
            category_indices,
        ):

            customer_index = int(
                customer_index
            )

            node_index = int(
                node_index
            )

            raw_value = str(
                dataframe.iloc[
                    customer_index
                ][category]
            )

            mapping_values[node_index].add(
                raw_value
            )

        # ----------------------------------------------------
        # Verify every graph node has exactly one category
        # ----------------------------------------------------

        node_count = graph[
            category
        ].num_nodes

        mapping = {}

        for node_index in range(
            node_count
        ):

            values = mapping_values.get(
                node_index,
                set(),
            )

            if len(values) == 0:
                raise ValueError(
                    f"{category}: node {node_index} "
                    "has no connected category value."
                )

            if len(values) > 1:
                raise ValueError(
                    f"{category}: node {node_index} "
                    f"maps to multiple values: {values}"
                )

            mapping[node_index] = (
                next(iter(values))
            )

        # ----------------------------------------------------
        # Verify reverse uniqueness
        # ----------------------------------------------------

        reverse = defaultdict(list)

        for node_index, value in mapping.items():
            reverse[value].append(
                node_index
            )

        duplicate_values = {
            value: indices
            for value, indices in reverse.items()
            if len(indices) != 1
        }

        if duplicate_values:
            raise ValueError(
                f"{category}: category values map to "
                f"multiple graph nodes: "
                f"{duplicate_values}"
            )

        # ----------------------------------------------------
        # Print verified mapping
        # ----------------------------------------------------

        print(
            f"\nVerified {category} mapping:"
        )

        for node_index, value in mapping.items():

            print(
                f"  node {node_index:>2} "
                f"→ {value}"
            )

        all_mappings[category] = mapping

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("MAPPING VERIFICATION")
    print("=" * 90)

    for category, mapping in all_mappings.items():

        expected_nodes = graph[
            category
        ].num_nodes

        assert len(mapping) == (
            expected_nodes
        )

        print(
            f"{category:<12} "
            f"{len(mapping)} verified nodes"
        )

    print("=" * 90)
    print(
        "GRAPH CATEGORY MAPPING VERIFICATION PASSED"
    )
    print("=" * 90)

    print(
        "\nThe mappings were recovered from the "
        "saved customer→category edges."
    )

    print(
        "No categorical node ordering was guessed."
    )

    print(
        "The graph itself was not modified."
    )


if __name__ == "__main__":
    main()