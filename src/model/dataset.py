"""
Olympus Graph – Dataset Builder
Converts Neo4j temporal snapshots into PyTorch Geometric HeteroData objects.

CRITICAL: The snapshot for year Y excludes ALL edges from year >= Y.
This ensures no data leakage during training/evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from loguru import logger

from src.config import DATA_PROCESSED_DIR, EMBEDDING_DIM
from src.graph.snapshot import (
    get_snapshot_athletes,
    get_snapshot_medal_edges,
    get_snapshot_competed_edges,
    get_snapshot_participated_edges,
    get_snapshot_represents_edges,
    get_events,
    get_countries,
    get_target_year_edges,
)
from src.utils import timed


# ── ID Mapping Helpers ───────────────────────────────

def build_id_map(ids: list[str]) -> dict[str, int]:
    """Create a string-ID → integer-index mapping."""
    return {str_id: idx for idx, str_id in enumerate(sorted(set(ids)))}


# ── Feature Construction ─────────────────────────────

def build_athlete_features(
    athletes: list[dict], id_map: dict[str, int], reference_year: int
) -> torch.Tensor:
    """
    Build athlete feature tensor.
    Features:
      [avg_perf_norm, max_perf_norm, num_participations_norm, num_games_norm,
       age_norm, height_norm, weight_norm, sex_binary, last_year_norm]
    """
    n = len(id_map)
    features = torch.zeros(n, 9)

    avg_vals = [float(a.get("avg_performance", 0) or 0) for a in athletes]
    max_vals = [float(a.get("max_performance", 0) or 0) for a in athletes]
    num_part_vals = [float(a.get("num_participations", 0) or 0) for a in athletes]
    num_games_vals = [float(a.get("num_games", 0) or 0) for a in athletes]

    max_avg = max(avg_vals) if avg_vals else 1.0
    max_max = max(max_vals) if max_vals else 1.0
    max_num_part = max(num_part_vals) if num_part_vals else 1.0
    max_num_games = max(num_games_vals) if num_games_vals else 1.0
    year_span = max(reference_year - 1896, 1)

    for a in athletes:
        idx = id_map.get(a["athlete_id"])
        if idx is None:
            continue

        avg_perf = float(a.get("avg_performance", 0) or 0)
        max_perf = float(a.get("max_performance", 0) or 0)
        num_part = float(a.get("num_participations", 0) or 0)
        num_games = float(a.get("num_games", 0) or 0)
        birth_year = float(a.get("birth_year", reference_year - 25) or (reference_year - 25))
        height = float(a.get("height", 175) or 175)
        weight = float(a.get("weight", 72) or 72)
        last_year = float(a.get("last_year", reference_year - 4) or (reference_year - 4))
        sex = 1.0 if a.get("sex") == "M" else 0.0

        # Rough age normalization relative to the current snapshot.
        age = (reference_year - 1) - birth_year
        age_norm = max(0.0, min(age / 100.0, 1.0))
        last_year_norm = max(0.0, min((last_year - 1896.0) / year_span, 1.0))

        features[idx] = torch.tensor([
            avg_perf / max_avg if max_avg > 0 else 0.0,
            max_perf / max_max if max_max > 0 else 0.0,
            num_part / max_num_part if max_num_part > 0 else 0.0,
            num_games / max_num_games if max_num_games > 0 else 0.0,
            age_norm,
            max(0.0, min(height / 220.0, 1.25)),
            max(0.0, min(weight / 200.0, 1.25)),
            sex,
            last_year_norm,
        ])

    return features


def build_country_features(
    countries: list[dict], id_map: dict[str, int], host_noc: str | None = None
) -> torch.Tensor:
    """
    Build country feature tensor.
    Features: [gdp_norm, population_norm, is_host]
    """
    n = len(id_map)
    features = torch.zeros(n, 3)

    # Collect GDP values for normalization
    gdps = [float(c.get("gdp", 0) or 0) for c in countries]
    pops = [float(c.get("population", 0) or 0) for c in countries]
    max_gdp = max(gdps) if gdps else 1
    max_pop = max(pops) if pops else 1

    for c in countries:
        idx = id_map.get(c["noc"])
        if idx is None:
            continue

        gdp = float(c.get("gdp", 0) or 0)
        pop = float(c.get("population", 0) or 0)
        is_host = 1.0 if c["noc"] == host_noc else 0.0

        features[idx] = torch.tensor([
            gdp / max_gdp if max_gdp > 0 else 0,
            pop / max_pop if max_pop > 0 else 0,
            is_host,
        ])

    return features


def build_event_features(
    events: list[dict], id_map: dict[str, int]
) -> torch.Tensor:
    """
    Build event feature tensor from pre-computed embeddings.
    Falls back to random if embeddings not available.
    """
    n = len(id_map)
    features = torch.zeros(n, EMBEDDING_DIM)

    # Try loading pre-computed embeddings
    emb_path = DATA_PROCESSED_DIR / "event_embeddings.parquet"
    emb_lookup = {}
    if emb_path.exists():
        emb_df = pd.read_parquet(emb_path)
        for _, row in emb_df.iterrows():
            emb_lookup[row["event_id"]] = row["embedding"]

    for ev in events:
        idx = id_map.get(ev["event_id"])
        if idx is None:
            continue

        if ev["event_id"] in emb_lookup:
            emb = emb_lookup[ev["event_id"]]
            features[idx] = torch.tensor(emb, dtype=torch.float)
        elif ev.get("embedding") is not None:
            features[idx] = torch.tensor(ev["embedding"], dtype=torch.float)
        else:
            # Random fallback
            features[idx] = torch.randn(EMBEDDING_DIM) * 0.1

    return features


def build_games_features(
    games_ids: list[str], id_map: dict[str, int]
) -> torch.Tensor:
    """
    Build Games feature tensor.
    Features: [year_norm, is_summer]
    """
    n = len(id_map)
    features = torch.zeros(n, 2)

    for gid in games_ids:
        idx = id_map.get(gid)
        if idx is None:
            continue
        parts = gid.split(" ")
        year = int(parts[0]) if parts else 2000
        is_summer = 1.0 if len(parts) > 1 and parts[1] == "Summer" else 0.0
        features[idx] = torch.tensor([year / 2024.0, is_summer])

    return features


# ── Edge Index Construction ──────────────────────────

def build_edge_index(
    edges: list[dict],
    src_key: str,
    dst_key: str,
    src_map: dict[str, int],
    dst_map: dict[str, int],
) -> torch.Tensor:
    """Build a (2, num_edges) edge index tensor from edge records."""
    src_indices = []
    dst_indices = []

    for e in edges:
        src_id = e[src_key]
        dst_id = e[dst_key]
        if src_id in src_map and dst_id in dst_map:
            src_indices.append(src_map[src_id])
            dst_indices.append(dst_map[dst_id])

    if not src_indices:
        return torch.zeros(2, 0, dtype=torch.long)

    return torch.tensor([src_indices, dst_indices], dtype=torch.long)


# ── Negative Sampling ────────────────────────────────

def sample_negative_edges(
    pos_edge_index: torch.Tensor,
    num_athletes: int,
    num_events: int,
    num_neg: int | None = None,
    competed_edge_index: torch.Tensor | None = None,
    hard_negative_ratio: float = 0.7,
) -> torch.Tensor:
    """
    Sample negative (Athlete, Event) pairs that are NOT in pos_edge_index.
    Mixes:
      1) hard negatives from COMPETED_IN but not WON_MEDAL
      2) random negatives across athlete-event pairs
    Returns (2, num_neg) tensor.
    """
    if num_neg is None:
        num_neg = pos_edge_index.size(1)  # 1:1 ratio

    pos_set = set()
    for i in range(pos_edge_index.size(1)):
        pos_set.add((pos_edge_index[0, i].item(), pos_edge_index[1, i].item()))

    selected_neg = set()
    neg_src: list[int] = []
    neg_dst: list[int] = []

    # Hard negatives: athletes who competed in an event but did not medal in it.
    if (
        competed_edge_index is not None
        and competed_edge_index.size(1) > 0
        and hard_negative_ratio > 0
    ):
        hard_pool = []
        for i in range(competed_edge_index.size(1)):
            pair = (competed_edge_index[0, i].item(), competed_edge_index[1, i].item())
            if pair not in pos_set:
                hard_pool.append(pair)

        np.random.shuffle(hard_pool)
        hard_target = min(int(num_neg * hard_negative_ratio), len(hard_pool))
        for s, d in hard_pool[:hard_target]:
            if (s, d) in selected_neg:
                continue
            selected_neg.add((s, d))
            neg_src.append(s)
            neg_dst.append(d)

    attempts = 0
    max_attempts = max(num_neg * 20, 10000)

    while len(neg_src) < num_neg and attempts < max_attempts:
        s = np.random.randint(0, num_athletes)
        d = np.random.randint(0, num_events)
        if (s, d) not in pos_set and (s, d) not in selected_neg:
            neg_src.append(s)
            neg_dst.append(d)
            selected_neg.add((s, d))
        attempts += 1

    return torch.tensor([neg_src, neg_dst], dtype=torch.long)


# ── Main Dataset Builder ─────────────────────────────

@timed
def build_hetero_data(
    max_year: int,
    host_noc: str | None = None,
) -> tuple[HeteroData, dict[str, dict[str, int]]]:
    """
    Build a PyG HeteroData object from a temporal snapshot up to max_year.

    Returns:
        (data, id_maps) where id_maps = {node_type: {string_id: int_index}}
    """
    logger.info(f"Building HeteroData snapshot for years < {max_year}")

    # ── Fetch data from Neo4j ─────────────────────
    athletes = get_snapshot_athletes(max_year)
    medal_edges = get_snapshot_medal_edges(max_year)
    competed_edges = get_snapshot_competed_edges(max_year)
    participated_edges = get_snapshot_participated_edges(max_year)
    represents_edges = get_snapshot_represents_edges()
    events = get_events()
    countries = get_countries()

    # ── Build ID maps ─────────────────────────────
    athlete_ids = [a["athlete_id"] for a in athletes]
    event_ids = [e["event_id"] for e in events]
    country_ids = [c["noc"] for c in countries]

    games_ids = [edge["games_id"] for edge in participated_edges]

    athlete_map = build_id_map(athlete_ids)
    event_map = build_id_map(event_ids)
    country_map = build_id_map(country_ids)
    games_map = build_id_map(games_ids)

    id_maps = {
        "athlete": athlete_map,
        "event": event_map,
        "country": country_map,
        "games": games_map,
    }

    # ── Build HeteroData ──────────────────────────
    data = HeteroData()

    # Node features
    data["athlete"].x = build_athlete_features(athletes, athlete_map, max_year)
    data["athlete"].num_nodes = len(athlete_map)

    data["country"].x = build_country_features(countries, country_map, host_noc)
    data["country"].num_nodes = len(country_map)

    data["event"].x = build_event_features(events, event_map)
    data["event"].num_nodes = len(event_map)

    data["games"].x = build_games_features(games_ids, games_map)
    data["games"].num_nodes = len(games_map)

    # Edge indices
    # COMPETED_IN: Athlete → Event
    data["athlete", "competed_in", "event"].edge_index = build_edge_index(
        competed_edges, "athlete_id", "event_id", athlete_map, event_map
    )

    # REPRESENTS: Athlete → Country
    data["athlete", "represents", "country"].edge_index = build_edge_index(
        represents_edges, "athlete_id", "noc", athlete_map, country_map
    )

    # PARTICIPATED_IN: Athlete → Games
    data["athlete", "participated_in", "games"].edge_index = build_edge_index(
        participated_edges, "athlete_id", "games_id", athlete_map, games_map
    )

    # WON_MEDAL: Athlete → Event (positive labels for link prediction)
    data["athlete", "won_medal", "event"].edge_index = build_edge_index(
        medal_edges, "athlete_id", "event_id", athlete_map, event_map
    )

    # Also add reverse edges for message passing
    for edge_type in [
        ("athlete", "competed_in", "event"),
        ("athlete", "represents", "country"),
        ("athlete", "participated_in", "games"),
    ]:
        src, rel, dst = edge_type
        rev_rel = f"rev_{rel}"
        forward_ei = data[edge_type].edge_index
        if forward_ei.size(1) > 0:
            data[dst, rev_rel, src].edge_index = forward_ei.flip(0)

    logger.info(
        f"HeteroData built: "
        f"{data['athlete'].num_nodes} athletes, "
        f"{data['event'].num_nodes} events, "
        f"{data['country'].num_nodes} countries, "
        f"WON_MEDAL edges: {data['athlete', 'won_medal', 'event'].edge_index.size(1)}"
    )

    return data, id_maps


@timed
def build_train_test_split(
    train_max_year: int = 2024,
    test_year: int = 2024,
    host_noc: str | None = None,
) -> dict:
    """
    Build train and test HeteroData objects.

    Train: all data before train_max_year
    Test:  ground truth medals from test_year
    """
    # Train graph
    train_data, train_maps = build_hetero_data(train_max_year, host_noc)

    # Test ground truth
    test_medals = get_target_year_edges(test_year)

    # Build test edge index using train's ID maps
    test_pos_src = []
    test_pos_dst = []
    test_medal_types = []

    for m in test_medals:
        a_idx = train_maps["athlete"].get(m["athlete_id"])
        e_idx = train_maps["event"].get(m["event_id"])
        if a_idx is not None and e_idx is not None:
            test_pos_src.append(a_idx)
            test_pos_dst.append(e_idx)
            test_medal_types.append(m["medal"])

    test_pos_edge_index = torch.tensor(
        [test_pos_src, test_pos_dst], dtype=torch.long
    ) if test_pos_src else torch.zeros(2, 0, dtype=torch.long)

    # Negative sampling for training
    train_pos_ei = train_data["athlete", "won_medal", "event"].edge_index
    train_neg_ei = sample_negative_edges(
        train_pos_ei,
        num_athletes=train_data["athlete"].num_nodes,
        num_events=train_data["event"].num_nodes,
        competed_edge_index=train_data["athlete", "competed_in", "event"].edge_index,
    )

    logger.info(
        f"Train/Test split: "
        f"train_pos={train_pos_ei.size(1)}, "
        f"train_neg={train_neg_ei.size(1)}, "
        f"test_pos={test_pos_edge_index.size(1)}"
    )

    return {
        "train_data": train_data,
        "train_maps": train_maps,
        "train_pos_edge_index": train_pos_ei,
        "train_neg_edge_index": train_neg_ei,
        "test_pos_edge_index": test_pos_edge_index,
        "test_medal_types": test_medal_types,
    }
