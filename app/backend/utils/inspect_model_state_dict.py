"""
Inspect the frozen GraphSAGE state dictionary.

Diagnostic only.
Does not modify the checkpoint or model.
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
    print("FROZEN GRAPHSAGE STATE DICT")
    print("=" * 90)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    print(
        "Number of state-dict entries:",
        len(state_dict),
    )

    print()

    for key, tensor in state_dict.items():

        print(
            f"{key:<80} "
            f"shape={tuple(tensor.shape)}"
        )

    print()
    print("=" * 90)
    print("STATE DICT INSPECTION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()