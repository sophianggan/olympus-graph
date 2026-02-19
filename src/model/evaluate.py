"""
Olympus Graph – Evaluation Metrics
Custom Recall@K implementation for Olympic medal prediction.

Recall@K: For each event, did the actual Gold medalist appear
in the model's top-K predicted athletes?
"""

from __future__ import annotations

import torch
import numpy as np
from collections import defaultdict
from loguru import logger


def recall_at_k(
    predictions: dict[str, list[tuple[str, float]]],
    ground_truth: dict[str, str],
    k: int = 3,
) -> dict[str, float]:
    """
    Compute Recall@K for medal predictions.

    Args:
        predictions: {event_id: [(athlete_id, score), ...]} sorted by score desc
        ground_truth: {event_id: gold_medalist_athlete_id}
        k: number of top predictions to consider

    Returns:
        {
            "recall_at_k": float,   # Main metric
            "hits": int,            # Number of correct top-K predictions
            "total": int,           # Total events evaluated
            "per_event": dict       # Per-event results
        }
    """
    hits = 0
    total = 0
    per_event = {}

    for event_id, gold_athlete in ground_truth.items():
        if event_id not in predictions:
            continue

        top_k_athletes = [aid for aid, _ in predictions[event_id][:k]]
        hit = gold_athlete in top_k_athletes
        hits += int(hit)
        total += 1

        per_event[event_id] = {
            "gold_medalist": gold_athlete,
            "top_k": top_k_athletes,
            "hit": hit,
        }

    recall = hits / total if total > 0 else 0.0

    return {
        "recall_at_k": recall,
        "k": k,
        "hits": hits,
        "total": total,
        "per_event": per_event,
    }


def compute_predictions_per_event(
    athlete_embs: torch.Tensor,
    event_embs: torch.Tensor,
    model,
    athlete_id_map: dict[str, int],
    event_id_map: dict[str, int],
    candidate_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """
    Compute medal predictions for all (athlete, event) candidate pairs.

    Args:
        athlete_embs: (num_athletes, hidden_dim) athlete node embeddings
        event_embs:   (num_events, hidden_dim) event node embeddings
        model:        OlympusHeteroGNN with predict_link method
        athlete_id_map: {athlete_id: int_index}
        event_id_map:   {event_id: int_index}
        candidate_pairs: optional list of (athlete_id, event_id) to evaluate.
                         If None, evaluates all combinations.

    Returns:
        {event_id: [(athlete_id, probability), ...]} sorted descending
    """
    # Reverse maps
    idx_to_athlete = {v: k for k, v in athlete_id_map.items()}
    idx_to_event = {v: k for k, v in event_id_map.items()}

    predictions = defaultdict(list)

    model.eval()
    with torch.no_grad():
        if candidate_pairs is not None:
            # Evaluate specific pairs
            for athlete_id, event_id in candidate_pairs:
                a_idx = athlete_id_map.get(athlete_id)
                e_idx = event_id_map.get(event_id)
                if a_idx is not None and e_idx is not None:
                    a_emb = athlete_embs[a_idx].unsqueeze(0)
                    e_emb = event_embs[e_idx].unsqueeze(0)
                    prob = model.predict_link(a_emb, e_emb).item()
                    predictions[event_id].append((athlete_id, prob))
        else:
            # Evaluate all athletes for each event (expensive but complete)
            for e_idx in range(event_embs.size(0)):
                e_id = idx_to_event.get(e_idx)
                if e_id is None:
                    continue

                e_emb = event_embs[e_idx].unsqueeze(0).expand(
                    athlete_embs.size(0), -1
                )
                probs = model.predict_link(athlete_embs, e_emb)

                for a_idx in range(athlete_embs.size(0)):
                    a_id = idx_to_athlete.get(a_idx)
                    if a_id is not None:
                        predictions[e_id].append((a_id, probs[a_idx].item()))

    # Sort each event's predictions by probability (descending)
    for event_id in predictions:
        predictions[event_id].sort(key=lambda x: x[1], reverse=True)

    return dict(predictions)


def build_ground_truth_from_edges(
    test_pos_edge_index: torch.Tensor,
    medal_types: list[str],
    athlete_id_map: dict[str, int],
    event_id_map: dict[str, int],
) -> dict[str, str]:
    """
    Build {event_id: gold_medalist_athlete_id} from test edge data.
    Only considers Gold medals for Recall@K.
    """
    idx_to_athlete = {v: k for k, v in athlete_id_map.items()}
    idx_to_event = {v: k for k, v in event_id_map.items()}

    ground_truth = {}
    for i in range(test_pos_edge_index.size(1)):
        if medal_types[i] == "Gold":
            a_idx = test_pos_edge_index[0, i].item()
            e_idx = test_pos_edge_index[1, i].item()
            a_id = idx_to_athlete.get(a_idx)
            e_id = idx_to_event.get(e_idx)
            if a_id and e_id:
                ground_truth[e_id] = a_id

    return ground_truth


def evaluate_model(
    model,
    data,
    split_info: dict,
    k: int = 3,
) -> dict:
    """
    Full evaluation pipeline.

    Args:
        model: trained OlympusHeteroGNN
        data: HeteroData (train snapshot)
        split_info: dict from build_train_test_split
        k: top-K for recall

    Returns:
        Recall@K results dict
    """
    model.eval()
    with torch.no_grad():
        x_dict = model(data)

    athlete_embs = x_dict["athlete"]
    event_embs = x_dict["event"]

    # Build ground truth (Gold medalists only)
    gt = build_ground_truth_from_edges(
        split_info["test_pos_edge_index"],
        split_info["test_medal_types"],
        split_info["train_maps"]["athlete"],
        split_info["train_maps"]["event"],
    )

    # Compute predictions for candidate pairs
    # Use athletes who competed in each test event
    predictions = compute_predictions_per_event(
        athlete_embs=athlete_embs,
        event_embs=event_embs,
        model=model,
        athlete_id_map=split_info["train_maps"]["athlete"],
        event_id_map=split_info["train_maps"]["event"],
        candidate_pairs=None,  # Evaluate all for thorough eval
    )

    results = recall_at_k(predictions, gt, k=k)

    if results["total"] == 0:
        logger.warning(
            "No test events with Gold medalists were available in the evaluated split."
        )

    logger.info(
        f"Recall@{k}: {results['recall_at_k']:.4f} "
        f"({results['hits']}/{results['total']} events)"
    )

    return results
