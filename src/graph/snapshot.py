"""
Olympus Graph – Temporal Snapshot Queries
Provides functions to query the Neo4j graph with temporal filtering.

CRITICAL: When building a snapshot for year Y, we ONLY see edges where
relationship.year < Y. This prevents data leakage during GNN training.
"""

from __future__ import annotations

from typing import Any
from loguru import logger

from src.utils import get_neo4j_driver, run_cypher


def get_snapshot_athletes(max_year: int) -> list[dict[str, Any]]:
    """
    Get all athletes who participated in Games BEFORE max_year.
    Returns athlete features aggregated up to (not including) max_year.
    """
    query = """
    MATCH (a:Athlete)-[p:PARTICIPATED_IN]->(g:Games)
    WHERE g.year < $max_year
    WITH a,
         avg(p.performance) AS avg_performance,
         max(p.performance) AS max_performance,
         count(p)           AS num_participations,
         max(g.year)        AS last_year,
         collect(DISTINCT g.year) AS years_active
    RETURN
        a.athlete_id   AS athlete_id,
        a.name         AS name,
        a.sex          AS sex,
        a.birth_year   AS birth_year,
        a.height       AS height,
        a.weight       AS weight,
        avg_performance,
        max_performance,
        num_participations,
        last_year,
        size(years_active) AS num_games
    """
    return run_cypher(query, {"max_year": max_year})


def get_snapshot_medal_edges(max_year: int) -> list[dict[str, Any]]:
    """
    Get all WON_MEDAL edges where year < max_year.
    These are the POSITIVE labels for link prediction training.
    """
    query = """
    MATCH (a:Athlete)-[m:WON_MEDAL]->(e:Event)
    WHERE m.year < $max_year
    RETURN
        a.athlete_id AS athlete_id,
        e.event_id   AS event_id,
        m.medal      AS medal,
        m.year       AS year
    """
    return run_cypher(query, {"max_year": max_year})


def get_snapshot_competed_edges(max_year: int) -> list[dict[str, Any]]:
    """
    Get all COMPETED_IN edges where year < max_year.
    These represent ALL athlete-event participation (both medal and no-medal).
    """
    query = """
    MATCH (a:Athlete)-[c:COMPETED_IN]->(e:Event)
    WHERE c.year < $max_year
    RETURN
        a.athlete_id AS athlete_id,
        e.event_id   AS event_id,
        c.year       AS year
    """
    return run_cypher(query, {"max_year": max_year})


def get_snapshot_participated_edges(max_year: int) -> list[dict[str, Any]]:
    """
    Get PARTICIPATED_IN edges with concrete Games IDs before max_year.
    """
    query = """
    MATCH (a:Athlete)-[p:PARTICIPATED_IN]->(g:Games)
    WHERE g.year < $max_year
    RETURN
        a.athlete_id AS athlete_id,
        g.games_id   AS games_id,
        g.year       AS year,
        g.season     AS season
    """
    return run_cypher(query, {"max_year": max_year})


def get_snapshot_represents_edges() -> list[dict[str, Any]]:
    """Get all REPRESENTS edges (these are static, no temporal filtering)."""
    query = """
    MATCH (a:Athlete)-[:REPRESENTS]->(c:Country)
    RETURN
        a.athlete_id AS athlete_id,
        c.noc        AS noc
    """
    return run_cypher(query)


def get_events() -> list[dict[str, Any]]:
    """Get all Event nodes with their embeddings (if computed)."""
    query = """
    MATCH (e:Event)
    RETURN
        e.event_id  AS event_id,
        e.sport     AS sport,
        e.event     AS event,
        e.embedding AS embedding
    """
    return run_cypher(query)


def get_countries() -> list[dict[str, Any]]:
    """Get all Country nodes with properties."""
    query = """
    MATCH (c:Country)
    RETURN
        c.noc        AS noc,
        c.gdp        AS gdp,
        c.population AS population
    """
    return run_cypher(query)


def get_target_year_edges(year: int) -> list[dict[str, Any]]:
    """
    Get actual medal edges for a specific year (ground truth for evaluation).
    """
    query = """
    MATCH (a:Athlete)-[m:WON_MEDAL]->(e:Event)
    WHERE m.year = $year
    RETURN
        a.athlete_id AS athlete_id,
        e.event_id   AS event_id,
        m.medal      AS medal,
        m.year       AS year
    """
    return run_cypher(query, {"year": year})


def get_candidates_for_event(event_id: str, max_year: int) -> list[dict[str, Any]]:
    """
    Get candidate athletes for a specific event.
    Candidates = athletes who have competed in this event before max_year.
    """
    query = """
    MATCH (a:Athlete)-[c:COMPETED_IN]->(e:Event {event_id: $event_id})
    WHERE c.year < $max_year
    WITH DISTINCT a
    MATCH (a)-[p:PARTICIPATED_IN]->(g:Games)
    WHERE g.year < $max_year
    WITH a,
         avg(p.performance) AS avg_perf,
         max(g.year)        AS last_year,
         count(p)           AS num_games
    MATCH (a)-[:REPRESENTS]->(country:Country)
    RETURN
        a.athlete_id AS athlete_id,
        a.name       AS name,
        a.birth_year AS birth_year,
        avg_perf,
        last_year,
        num_games,
        country.noc  AS noc
    ORDER BY avg_perf DESC
    """
    return run_cypher(query, {"event_id": event_id, "max_year": max_year})


def get_athlete_neighborhood(athlete_id: str, max_year: int) -> dict[str, Any]:
    """
    Get the 1-hop neighborhood of an athlete for visualization.
    Returns nodes and edges respecting temporal constraints.
    """
    query = """
    MATCH (a:Athlete {athlete_id: $athlete_id})
    OPTIONAL MATCH (a)-[:REPRESENTS]->(c:Country)
    OPTIONAL MATCH (a)-[p:PARTICIPATED_IN]->(g:Games)
        WHERE g.year < $max_year
    OPTIONAL MATCH (a)-[comp:COMPETED_IN]->(e:Event)
        WHERE comp.year < $max_year
    OPTIONAL MATCH (a)-[m:WON_MEDAL]->(me:Event)
        WHERE m.year < $max_year
    RETURN
        a {.*, labels: labels(a)}                               AS athlete,
        collect(DISTINCT c {.*, labels: labels(c)})             AS countries,
        collect(DISTINCT g {.*, labels: labels(g)})             AS games,
        collect(DISTINCT e {.*, labels: labels(e)})             AS events,
        collect(DISTINCT {event: me.event_id, medal: m.medal, year: m.year}) AS medals
    """
    results = run_cypher(query, {"athlete_id": athlete_id, "max_year": max_year})
    return results[0] if results else {}
