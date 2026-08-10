"""
Inspect the saved heterogeneous graph for categorical
node-to-value mappings.

This script is diagnostic only.

It does NOT:
- modify the graph
- retrain the model
- create mappings
- guess categorical indices
"""

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]

GRAPH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "graph"
    / "bank_heterodata.pt"
)


CATEGORICAL_NODE_TYPES = [
    "job",
    "education",
    "marital",
    "contact",
    "month",
]


def load_graph():
    """Load the saved heterogeneous graph."""

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


def inspect_object_attributes(
    graph,
    node_type,
):
    """
    Inspect non-tensor attributes attached to a node store.
    """

    store = graph[node_type]

    print(
        f"\n[{node_type}] store attributes:"
    )

    for key in store.keys():

        value = store[key]

        print(
            f"  {key}: "
            f"type={type(value).__name__}"
        )

        if isinstance(value, list):
            print(
                f"    values={value}"
            )

        elif isinstance(value, dict):
            print(
                f"    keys={list(value.keys())}"
            )

        elif isinstance(value, str):
            print(
                f"    value={value}"
            )


def main():

    print("=" * 85)
    print("GRAPH CATEGORY MAPPING VERIFICATION")
    print("=" * 85)

    graph = load_graph()

    print(
        "Graph type:",
        type(graph).__name__,
    )

    print(
        "\nNode types:"
    )

    for node_type in graph.node_types:
        print(
            f"  {node_type}: "
            f"{graph[node_type].num_nodes} nodes"
        )

    print(
        "\nGraph-level attributes:"
    )

    graph_attributes = list(
        graph.keys()
    )

    if graph_attributes:
        for key in graph_attributes:
            try:
                value = graph[key]

                print(
                    f"  {key}: "
                    f"type={type(value).__name__}"
                )

            except Exception:
                pass

    else:
        print(
            "  No graph-level attributes found."
        )

    # --------------------------------------------------------
    # Inspect categorical node stores
    # --------------------------------------------------------

    for node_type in CATEGORICAL_NODE_TYPES:

        if node_type not in graph.node_types:
            print(
                f"\nWARNING: node type '{node_type}' "
                "is not present."
            )
            continue

        inspect_object_attributes(
            graph,
            node_type,
        )

    # --------------------------------------------------------
    # Search common mapping attribute names
    # --------------------------------------------------------

    print(
        "\nSearching for categorical mappings..."
    )

    mapping_names = [
        "mapping",
        "mappings",
        "category_mapping",
        "category_mappings",
        "categories",
        "values",
        "names",
        "classes",
        "class_names",
        "label_mapping",
        "value_to_index",
        "index_to_value",
    ]

    found = []

    for node_type in CATEGORICAL_NODE_TYPES:

        if node_type not in graph.node_types:
            continue

        store = graph[node_type]

        for name in mapping_names:

            if name not in store:
                continue

            value = store[name]

            found.append(
                (
                    node_type,
                    name,
                    value,
                )
            )

            print(
                f"\nFOUND: "
                f"{node_type}.{name}"
            )

            print(
                "Type:",
                type(value).__name__,
            )

            print(
                "Value:",
                value,
            )

    # --------------------------------------------------------
    # Inspect metadata if present
    # --------------------------------------------------------

    print(
        "\nChecking graph metadata..."
    )

    possible_metadata_names = [
        "metadata",
        "graph_metadata",
        "category_mappings",
        "mappings",
    ]

    for name in possible_metadata_names:

        if hasattr(graph, name):

            try:
                value = getattr(
                    graph,
                    name,
                )

                print(
                    f"\nFOUND graph attribute: "
                    f"{name}"
                )

                print(
                    value
                )

            except Exception as exc:

                print(
                    f"Could not read {name}: "
                    f"{exc}"
                )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 85)

    if found:

        print(
            "CATEGORY MAPPINGS FOUND"
        )

        print(
            "Do not modify them."
        )

        print(
            "They can be used by the "
            "production graph-inference service."
        )

    else:

        print(
            "NO EXPLICIT CATEGORY MAPPINGS FOUND"
        )

        print(
            "STOP: Do not guess categorical "
            "node indices."
        )

        print(
            "The mapping must be recovered from "
            "the graph-construction/preprocessing "
            "artifacts before raw CSV inference "
            "is enabled."
        )

    print("=" * 85)


if __name__ == "__main__":
    main()