"""
Olympus Graph – Agent Tools
Tools for the LangGraph self-correcting agent.

Tools:
  1. GraphQueryTool  — Text-to-Cypher, queries Neo4j for historical data
  2. ModelPredictTool — Runs GNN inference for medal predictions
"""

from __future__ import annotations

import json
from typing import Any
from loguru import logger

from src.utils import run_cypher, get_neo4j_driver
from src.graph.snapshot import (
    get_candidates_for_event,
    get_athlete_neighborhood,
    get_target_year_edges,
)


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
        from src.model.train import load_model
        from src.model.dataset import build_hetero_data
        from src.graph.preprocess import HOST_COUNTRY_MAP

        host_noc = HOST_COUNTRY_MAP.get(target_year)

        # Find matching event(s)
        event_search = run_cypher(
            """
            MATCH (e:Event)
            WHERE toLower(e.event) CONTAINS toLower($search)
               OR toLower(e.event_id) CONTAINS toLower($search)
            RETURN e.event_id AS event_id, e.event AS event
            LIMIT 5
            """,
            {"search": event_name},
        )

        if not event_search:
            return {
                "success": False,
                "predictions": [],
                "event": event_name,
                "error": f"No event matching '{event_name}' found in the graph.",
            }

        matched_event_id = event_search[0]["event_id"]
        matched_event_name = event_search[0]["event"]

        # Build snapshot and load model
        data, id_maps = build_hetero_data(max_year=target_year, host_noc=host_noc)
        model = load_model()

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

        # Score all athletes against this event
        e_emb = event_embs[event_idx].unsqueeze(0).expand(athlete_embs.size(0), -1)
        scores = model.predict_link(athlete_embs, e_emb)

        # Get top-K
        top_values, top_indices = torch.topk(scores, min(top_k, scores.size(0)))

        idx_to_athlete = {v: k for k, v in id_maps["athlete"].items()}

        predictions = []
        for i in range(top_values.size(0)):
            a_idx = top_indices[i].item()
            a_id = idx_to_athlete.get(a_idx, "Unknown")

            # Fetch athlete details
            details = run_cypher(
                """
                MATCH (a:Athlete {athlete_id: $aid})-[:REPRESENTS]->(c:Country)
                RETURN a.name AS name, a.birth_year AS birth_year, c.noc AS country
                """,
                {"aid": a_id},
            )
            name = details[0]["name"] if details else a_id.split(" | ")[0]
            country = details[0]["country"] if details else "N/A"
            birth_year = details[0].get("birth_year", 0) if details else 0

            predictions.append({
                "rank": i + 1,
                "athlete_id": a_id,
                "name": name,
                "country": country,
                "age_at_games": target_year - birth_year if birth_year else None,
                "probability": round(top_values[i].item(), 4),
            })

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
