"""Ablation study entry point; see experiments/ablation/ (python ablation.py --help)."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from experiments.ablation.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
