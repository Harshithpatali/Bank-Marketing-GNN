"""
Central configuration and artifact paths for the
Bank Marketing GraphSAGE backend.
"""

from pathlib import Path


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# Artifact directories
# ============================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR = ARTIFACTS_DIR / "models"
GRAPH_DIR = ARTIFACTS_DIR / "graph"
RESULTS_DIR = ARTIFACTS_DIR / "results"


# ============================================================
# Final frozen model
# ============================================================

FINAL_MODEL_PATH = (
    MODELS_DIR / "hetero_graphsage_final.pt"
)


# ============================================================
# Final heterogeneous graph
# ============================================================

GRAPH_PATH = (
    GRAPH_DIR / "bank_heterodata.pt"
)


# ============================================================
# Training-fitted preprocessing transformer
# ============================================================

SCALER_DIR = ARTIFACTS_DIR / "scaler"

PREPROCESSOR_PATH = (
    SCALER_DIR / "preprocessor.joblib"
)


# ============================================================
# Final evaluation artifacts
# ============================================================

FINAL_METRICS_PATH = (
    RESULTS_DIR
    / "final"
    / "final_test_metrics.json"
)

FINAL_PREDICTIONS_PATH = (
    RESULTS_DIR
    / "final"
    / "final_predictions.csv"
)


# ============================================================
# Explainability artifacts
# ============================================================

EXPLAINABILITY_DIR = (
    RESULTS_DIR / "explainability"
)


# ============================================================
# Runtime configuration
# ============================================================

API_TITLE = (
    "Bank Marketing GraphSAGE API"
)

API_DESCRIPTION = (
    "REST API for bank marketing subscription prediction "
    "using a frozen heterogeneous GraphSAGE model."
)

API_VERSION = "1.0.0"

CLASSIFICATION_THRESHOLD = 0.5


# ============================================================
# Required inference artifacts
# ============================================================

def validate_artifact_paths() -> None:
    """
    Validate that all artifacts required for production
    inference exist.
    """

    required_paths = {
        "final_model": FINAL_MODEL_PATH,
        "graph": GRAPH_PATH,
        "preprocessor": PREPROCESSOR_PATH,
    }

    missing = {
        name: str(path)
        for name, path in required_paths.items()
        if not path.exists()
    }

    if missing:
        details = "\n".join(
            f"- {name}: {path}"
            for name, path in missing.items()
        )

        raise FileNotFoundError(
            "Required backend artifacts are missing:\n"
            + details
        )