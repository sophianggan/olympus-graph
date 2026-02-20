"""
Olympus Graph – GNN Training Loop
Trains the Heterogeneous GraphSAGE/GATv2 for WON_MEDAL link prediction.

Training Strategy:
  - Train: Games before TEST_YEAR (from src.config)
  - Test:  TEST_YEAR hold-out (from src.config)
  - Loss:  Binary Cross Entropy (edge existence)
  - Metric: Recall@3
"""

from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from pathlib import Path
from loguru import logger

from src.config import (
    GNN_LEARNING_RATE,
    GNN_EPOCHS,
    GNN_HIDDEN_DIM,
    MODEL_CHECKPOINT_DIR,
    RECALL_K,
    TRAIN_YEARS,
    TEST_YEAR,
)
from src.model.gnn import OlympusHeteroGNN, build_model
from src.model.dataset import (
    build_train_test_split,
    sample_negative_edges,
)
from src.model.evaluate import evaluate_model
from src.utils import timed


# ── Training Step ────────────────────────────────────

def train_epoch(
    model: OlympusHeteroGNN,
    data,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Single training epoch. Returns loss value."""
    model.train()
    optimizer.zero_grad()

    loss = model.compute_loss(data, pos_edge_index, neg_edge_index)
    loss.backward()
    optimizer.step()

    return loss.item()


# ── Validation Step ──────────────────────────────────

@torch.no_grad()
def validate(
    model: OlympusHeteroGNN,
    data,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
) -> float:
    """Compute validation loss."""
    model.eval()
    loss = model.compute_loss(data, pos_edge_index, neg_edge_index)
    return loss.item()


# ── Full Training Pipeline ───────────────────────────

@timed
def train_model(
    epochs: int = GNN_EPOCHS,
    lr: float = GNN_LEARNING_RATE,
    eval_every: int = 10,
    host_noc: str | None = None,
) -> tuple[OlympusHeteroGNN, dict]:
    """
    Full training pipeline.

    Steps:
    1. Build train/test split from Neo4j snapshots
    2. Construct HeteroData
    3. Train GNN with BCE loss
    4. Evaluate with Recall@3 periodically
    5. Save best checkpoint

    Returns:
        (model, eval_results)
    """
    logger.info("=" * 60)
    logger.info("OLYMPUS GRAPH — GNN Training")
    logger.info("=" * 60)

    # ── Build dataset ─────────────────────────────
    split_info = build_train_test_split(
        train_max_year=max(TRAIN_YEARS) + 1,  # sees edges with year < train_max_year
        test_year=TEST_YEAR,
        host_noc=host_noc,
    )

    data = split_info["train_data"]
    pos_ei = split_info["train_pos_edge_index"]
    neg_ei = split_info["train_neg_edge_index"]

    logger.info(f"Graph metadata: {data.metadata()}")

    # ── Build model ───────────────────────────────
    model = build_model(metadata=data.metadata())
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ─────────────────────────────
    best_loss = float("inf")
    best_recall = -1.0

    MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Re-sample negatives every epoch for diversity
        if epoch % 5 == 0:
            neg_ei = sample_negative_edges(
                pos_ei,
                num_athletes=data["athlete"].num_nodes,
                num_events=data["event"].num_nodes,
                competed_edge_index=data["athlete", "competed_in", "event"].edge_index,
            )

        loss = train_epoch(model, data, pos_ei, neg_ei, optimizer)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Loss: {loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        # Periodic evaluation
        if epoch % eval_every == 0 or epoch == epochs:
            eval_results = evaluate_model(
                model, data, split_info, k=RECALL_K
            )

            recall = eval_results["recall_at_k"]
            if recall > best_recall:
                best_recall = recall
                ckpt_path = MODEL_CHECKPOINT_DIR / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "recall_at_k": recall,
                    "metadata": data.metadata(),
                }, ckpt_path)
                logger.success(
                    f"New best Recall@{RECALL_K}: {recall:.4f} "
                    f"(saved to {ckpt_path})"
                )

        # Early stopping on loss
        if loss < best_loss:
            best_loss = loss

    # ── Final evaluation ──────────────────────────
    logger.info("=" * 40)
    logger.info("Final Evaluation")
    logger.info("=" * 40)

    final_results = evaluate_model(model, data, split_info, k=RECALL_K)

    # Save final model
    final_path = MODEL_CHECKPOINT_DIR / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "metadata": data.metadata(),
        "eval_results": final_results,
    }, final_path)
    logger.info(f"Final model saved to {final_path}")

    return model, final_results


# ── Load Trained Model ───────────────────────────────

def load_model(
    checkpoint_path: str | Path | None = None,
) -> OlympusHeteroGNN:
    """Load a trained model from checkpoint."""
    if checkpoint_path is None:
        checkpoint_path = MODEL_CHECKPOINT_DIR / "best_model.pt"

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = ckpt["metadata"]

    model = build_model(metadata=metadata)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch. Retrain with "
            "`python -m src.model.train` after updating model/data features."
        ) from exc
    model.eval()

    logger.info(
        f"Loaded model from {checkpoint_path} "
        f"(Recall@K={ckpt.get('recall_at_k', 'N/A')})"
    )

    return model


# ── CLI Entry Point ──────────────────────────────────

if __name__ == "__main__":
    model, results = train_model()
    logger.info(f"Training complete. Final Recall@{RECALL_K}: {results['recall_at_k']:.4f}")
