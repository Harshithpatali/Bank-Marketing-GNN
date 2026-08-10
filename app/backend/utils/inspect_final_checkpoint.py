"""
Inspect the frozen GraphSAGE checkpoint.

Diagnostic only.
Does not modify or retrain anything.
"""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "hetero_graphsage_final.pt"
)


def main():

    print("=" * 90)
    print("FINAL CHECKPOINT INSPECTION")
    print("=" * 90)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    print(
        "Checkpoint type:",
        type(checkpoint).__name__,
    )

    if isinstance(
        checkpoint,
        dict,
    ):

        print("\nCheckpoint keys:")

        for key in checkpoint.keys():

            value = checkpoint[key]

            print(
                f"  {key}: "
                f"{type(value).__name__}"
            )

        print("\nModel config:")

        print(
            checkpoint.get(
                "model_config"
            )
        )

        print("\nFreeze status:")

        print(
            checkpoint.get(
                "freeze_status"
            )
        )

        print("\nGraph metadata:")

        print(
            checkpoint.get(
                "graph_metadata"
            )
        )

        print("\nEdge types:")

        print(
            checkpoint.get(
                "edge_types"
            )
        )

        state_dict = checkpoint.get(
            "model_state_dict"
        )

        if state_dict is not None:

            print(
                "\nState-dict keys:"
            )

            for key in state_dict.keys():

                tensor = state_dict[key]

                print(
                    f"  {key}: "
                    f"{tuple(tensor.shape)}"
                )

    else:

        print(
            "\nCheckpoint is not a dictionary."
        )

        print(
            "Checkpoint contents:"
        )

        print(
            checkpoint
        )

    print("=" * 90)
    print(
        "CHECKPOINT INSPECTION COMPLETE"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()