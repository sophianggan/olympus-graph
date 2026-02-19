"""
Olympus Graph – Agent Tools
Tools for the LangGraph self-correcting agent.

Tools:
  1. GraphQueryTool  — Text-to-Cypher, queries Neo4j for historical data
  2. ModelPredictTool — Runs GNN inference for medal predictions
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any
from loguru import logger

from src.utils import run_cypher, get_neo4j_driver
from src.graph.snapshot import (
    get_candidates_for_event,
    get_athlete_neighborhood,
    get_target_year_edges,
)

_SNAPSHOT_CACHE: dict[tuple[int, str | None], tuple[Any, dict[str, dict[str, int]]]] = {}
_MODEL_CACHE = None
_ATHLETE_DETAILS_CACHE: dict[str, dict[str, Any]] = {}

_EVENT_STOPWORDS = {
    "the",
    "a",
    "an",
    "who",
    "will",
    "win",
    "winner",
    "predict",
    "prediction",
    "for",
    "in",
    "of",
    "olympics",
    "olympic",
    "event",
}


def _normalize_event_text(text: str) -> str:
    """Normalize free-text event queries and event names for robust matching."""
    value = (text or "").lower()
    value = value.replace("’", "'").replace("`", "'")
    value = re.sub(r"\b(\d{2,4})\s*m\b", r"\1 metres", value)
    value = re.sub(r"\b(\d{2,4})m\b", r"\1 metres", value)
    value = value.replace("meter", "metre")
    value = value.replace("metreses", "metres")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _is_track_sprint_query(normalized_query: str) -> bool:
    """Heuristic: short sprint track event where users commonly write 100m/200m."""
    if "metres" not in normalized_query:
        return False
    has_distance = any(distance in normalized_query for distance in ("100", "200", "400"))
    has_swim_terms = any(
        term in normalized_query
        for term in ("swimming", "freestyle", "backstroke", "breaststroke", "butterfly")
    )
    return has_distance and not has_swim_terms


@lru_cache(maxsize=1)
def _get_event_catalog() -> list[dict[str, Any]]:
    """Load all events once for fuzzy matching."""
    return run_cypher(
        """
        MATCH (e:Event)
        RETURN e.event_id AS event_id, e.event AS event, e.sport AS sport
        """
    )


def _find_best_event_match(event_name: str) -> dict[str, Any] | None:
    """Return the best matching event row for a user-provided event name."""
    normalized_query = _normalize_event_text(event_name)
    query_tokens = {
        token for token in normalized_query.split() if token not in _EVENT_STOPWORDS
    }
    if not query_tokens:
        return None

    best_row = None
    best_score = -1

    for row in _get_event_catalog():
        event_blob = f"{row.get('sport', '')} {row.get('event', '')} {row.get('event_id', '')}"
        normalized_event = _normalize_event_text(event_blob)
        event_tokens = set(normalized_event.split())

        score = 0
        if normalized_query and normalized_query in normalized_event:
            score += 12

        overlap = len(query_tokens & event_tokens)
        score += overlap * 3

        sport = str(row.get("sport", "")).lower()
        if "athletics" in query_tokens and sport == "athletics":
            score += 4
        if "swimming" in query_tokens and sport == "swimming":
            score += 4
        if _is_track_sprint_query(normalized_query) and sport == "athletics":
            score += 5

        if "men" in query_tokens and "men" in event_tokens:
            score += 2
        if "women" in query_tokens and "women" in event_tokens:
            score += 2

        if score > best_score:
            best_score = score
            best_row = row

    # Avoid random bad matches when overlap is weak.
    return best_row if best_score >= 4 else None


def _get_cached_model():
    """Load trained model once per process."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        from src.model.train import load_model

        _MODEL_CACHE = load_model()
    return _MODEL_CACHE


def _get_cached_snapshot(target_year: int, host_noc: str | None):
    """Build and cache the hetero snapshot per target year."""
    key = (target_year, host_noc)
    if key not in _SNAPSHOT_CACHE:
        from src.model.dataset import build_hetero_data

        _SNAPSHOT_CACHE[key] = build_hetero_data(max_year=target_year, host_noc=host_noc)
    return _SNAPSHOT_CACHE[key]


def _get_cached_athlete_details(athlete_id: str) -> dict[str, Any]:
    """Cache athlete detail lookups used in prediction formatting."""
    if athlete_id not in _ATHLETE_DETAILS_CACHE:
        details = run_cypher(
            """
            MATCH (a:Athlete {athlete_id: $aid})-[:REPRESENTS]->(c:Country)
            RETURN a.name AS name, a.birth_year AS birth_year, c.noc AS country
            """,
            {"aid": athlete_id},
        )
        _ATHLETE_DETAILS_CACHE[athlete_id] = details[0] if details else {}
    return _ATHLETE_DETAILS_CACHE[athlete_id]


# ══════════════════════════════════════════════════════
# Tool 1: GraphQueryTool (Text-to-Cypher)
# ══════════════════════════════════════════════════════

# Schema description for the LLM to generate Cypher
GRAPH_SCHEMA_DESCRIPTION = """
Neo4j Graph Schema:

NODE TYPES:
- (:Athlete {athlete_id, name, sex, birth_year, height, weight})
- (:Country {noc, gdp, population})
- (:Event {event_id, sport, event, embedding})
- (:Games {games_id, year, season, city, host_noc})

RELATIONSHIP TYPES:
- (:Athlete)-[:PARTICIPATED_IN {year, age, performance}]->(:Games)
- (:Athlete)-[:REPRESENTS]->(:Country)
- (:Athlete)-[:COMPETED_IN {year}]->(:Event)
- (:Athlete)-[:WON_MEDAL {year, medal}]->(:Event)
  medal is one of: 'Gold', 'Silver', 'Bronze'

IMPORTANT:
- Event IDs are formatted as "Sport | Event Name" (e.g., "Athletics | Athletics Men's 100 metres")
- Games IDs are formatted as "Year Season" (e.g., "2020 Summer")
- Use year properties on relationships for temporal filtering
"""


def graph_query_tool(cypher_query: str) -> dict[str, Any]:
    """
    Execute a Cypher query against the Neo4j knowledge graph.

    Args:
        cypher_query: A valid Cypher query string

    Returns:
        {"success": bool, "data": list[dict] | str, "row_count": int}
    """
    try:
        logger.info(f"Executing Cypher: {cypher_query[:100]}...")
        results = run_cypher(cypher_query)
        logger.info(f"Query returned {len(results)} rows")
        return {
            "success": True,
            "data": results,
            "row_count": len(results),
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Cypher execution failed: {error_msg}")
        return {
            "success": False,
            "data": error_msg,
            "row_count": 0,
        }


# ══════════════════════════════════════════════════════
# Tool 2: ModelPredictTool (GNN Inference)
# ══════════════════════════════════════════════════════

def model_predict_tool(
    event_name: str,
    target_year: int = 2028,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Run GNN inference to predict top medal candidates for an event.

    Args:
        event_name: Event description (e.g., "Men's 100 metres")
        target_year: Year to predict for (default 2028)
        top_k: Number of top predictions to return

    Returns:
        {"success": bool, "predictions": list[dict], "event": str}
    """
    try:
        import torch
        from src.graph.preprocess import HOST_COUNTRY_MAP

        host_noc = HOST_COUNTRY_MAP.get(target_year)

        # Find best matching event using normalized fuzzy matching.
        matched_event = _find_best_event_match(event_name)
        if not matched_event:
            return {
                "success": False,
                "predictions": [],
                "event": event_name,
                "error": f"No event matching '{event_name}' found in the graph.",
            }

        matched_event_id = matched_event["event_id"]
        matched_event_name = matched_event["event"]

        # Build snapshot and load model (cached to avoid repeated heavy rebuilds).
        data, id_maps = _get_cached_snapshot(target_year=target_year, host_noc=host_noc)
        model = _get_cached_model()

        # Get embeddings
        model.eval()
        with torch.no_grad():
            x_dict = model(data)

        athlete_embs = x_dict["athlete"]
        event_embs = x_dict["event"]

        event_idx = id_maps["event"].get(matched_event_id)
        if event_idx is None:
            return {
                "success": False,
                "predictions": [],
                "event": matched_event_name,
                "error": "Event not found in model's index.",
            }

        # Candidate filtering:
        # 1) must have historical participation in this event
        # 2) plausible competitive age at target year (14-45)
        candidates = get_candidates_for_event(matched_event_id, max_year=target_year)
        strict_indices = []
        relaxed_indices = []

        min_age = 14
        strict_max_age = 40
        relaxed_max_age = 45
        max_inactive_years = 12

        for candidate in candidates:
            athlete_id = candidate.get("athlete_id")
            if not athlete_id:
                continue
            idx = id_maps["athlete"].get(athlete_id)
            if idx is None:
                continue

            birth_year = candidate.get("birth_year")
            if birth_year is None:
                continue
            age_at_target = target_year - int(birth_year)
            if not (min_age <= age_at_target <= relaxed_max_age):
                continue

            relaxed_indices.append(idx)

            last_year = candidate.get("last_year")
            if last_year is None:
                continue
            years_since_last_games = target_year - int(last_year)
            if min_age <= age_at_target <= strict_max_age and years_since_last_games <= max_inactive_years:
                strict_indices.append(idx)

        candidate_indices = strict_indices if len(strict_indices) >= top_k else relaxed_indices

        # Fallback: if filtering removed everything, use all athletes.
        # Score a wider pool than top_k so deduping still leaves enough rows.
        if candidate_indices:
            candidate_tensor = torch.tensor(sorted(set(candidate_indices)), dtype=torch.long)
            athlete_pool = athlete_embs[candidate_tensor]
            e_emb = event_embs[event_idx].unsqueeze(0).expand(athlete_pool.size(0), -1)
            scores = model.predict_link(athlete_pool, e_emb)
            candidate_limit = min(max(top_k * 8, top_k + 5), scores.size(0))
            selected_scores, top_local_indices = torch.topk(scores, candidate_limit)
            selected_indices = candidate_tensor[top_local_indices]
        else:
            e_emb = event_embs[event_idx].unsqueeze(0).expand(athlete_embs.size(0), -1)
            scores = model.predict_link(athlete_embs, e_emb)
            candidate_limit = min(max(top_k * 8, top_k + 5), scores.size(0))
            selected_scores, selected_indices = torch.topk(scores, candidate_limit)

        idx_to_athlete = {v: k for k, v in id_maps["athlete"].items()}

        predictions = []
        for i in range(selected_scores.size(0)):
            a_idx = selected_indices[i].item()
            a_id = idx_to_athlete.get(a_idx, "Unknown")

            details = _get_cached_athlete_details(a_id)
            name = details.get("name", a_id)
            country = details.get("country", "N/A")
            birth_year = details.get("birth_year", 0)

            predictions.append({
                "rank": i + 1,
                "athlete_id": a_id,
                "name": name,
                "country": country,
                "age_at_games": target_year - birth_year if birth_year else None,
                "probability": round(selected_scores[i].item(), 4),
            })

        # Deduplicate by (name, country) and keep the highest probability variant.
        dedup: dict[tuple[str, str], dict[str, Any]] = {}
        for row in predictions:
            key = (str(row.get("name", "")).strip().lower(), str(row.get("country", "")))
            existing = dedup.get(key)
            if existing is None or row["probability"] > existing["probability"]:
                dedup[key] = row

        predictions = sorted(
            dedup.values(),
            key=lambda r: r["probability"],
            reverse=True,
        )[:top_k]

        for i, row in enumerate(predictions, start=1):
            row["rank"] = i

        logger.info(
            f"Predictions for {matched_event_name} ({target_year}): "
            f"{[p['name'] for p in predictions]}"
        )

        return {
            "success": True,
            "predictions": predictions,
            "event": matched_event_name,
            "event_id": matched_event_id,
            "target_year": target_year,
            "host_country": host_noc,
        }

    except FileNotFoundError:
        return {
            "success": False,
            "predictions": [],
            "event": event_name,
            "error": "Model checkpoint not found. Please train the model first (python -m src.model.train).",
        }
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        return {
            "success": False,
            "predictions": [],
            "event": event_name,
            "error": str(e),
        }


# ══════════════════════════════════════════════════════
# Tool Definitions (for LangGraph)
# ══════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    {
        "name": "graph_query",
        "description": (
            "Query the Olympic Knowledge Graph using Cypher. "
            "Use for historical data questions: past medalists, "
            "athlete statistics, country performance, etc. "
            f"\n\n{GRAPH_SCHEMA_DESCRIPTION}"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cypher_query": {
                    "type": "string",
                    "description": "A valid Cypher query for the Olympic graph.",
                },
            },
            "required": ["cypher_query"],
        },
    },
    {
        "name": "model_predict",
        "description": (
            "Predict future Olympic medalists using the GNN model. "
            "Use for future prediction questions: 'Who will win X in Y?' "
            "Returns top-K athletes with probabilities."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "event_name": {
                    "type": "string",
                    "description": "The event name (e.g., \"Men's 100 metres\").",
                },
                "target_year": {
                    "type": "integer",
                    "description": "The Olympic year to predict for (default 2028).",
                    "default": 2028,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top predictions (default 3).",
                    "default": 3,
                },
            },
            "required": ["event_name"],
        },
    },
]
