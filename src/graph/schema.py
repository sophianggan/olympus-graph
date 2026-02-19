"""
Olympus Graph – Neo4j Schema Setup
Creates constraints, indexes, and the core graph schema.

Node Types:
  - Athlete  (athlete_id = Name + DOB)
  - Country  (noc)
  - Event    (event_id = Sport + " | " + Event)
  - Games    (games_id = Year + " " + Season)

Edge Types:
  - PARTICIPATED_IN  (Athlete → Games)  {age, performance}
  - REPRESENTS       (Athlete → Country)
  - COMPETED_IN      (Athlete → Event)  {year}
  - WON_MEDAL        (Athlete → Event)  {medal, year}
"""

from loguru import logger
from src.utils import get_neo4j_driver


# ── Constraints & Indexes ────────────────────────────

SCHEMA_STATEMENTS = [
    # Uniqueness constraints (also create indexes)
    "CREATE CONSTRAINT athlete_id IF NOT EXISTS FOR (a:Athlete) REQUIRE a.athlete_id IS UNIQUE",
    "CREATE CONSTRAINT country_noc IF NOT EXISTS FOR (c:Country) REQUIRE c.noc IS UNIQUE",
    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
    "CREATE CONSTRAINT games_id IF NOT EXISTS FOR (g:Games) REQUIRE g.games_id IS UNIQUE",

    # Composite indexes for temporal queries
    "CREATE INDEX athlete_name_idx IF NOT EXISTS FOR (a:Athlete) ON (a.name)",
    "CREATE INDEX games_year_idx IF NOT EXISTS FOR (g:Games) ON (g.year)",
    "CREATE INDEX event_sport_idx IF NOT EXISTS FOR (e:Event) ON (e.sport)",
]


def create_schema():
    """Apply all constraints and indexes to Neo4j."""
    driver = get_neo4j_driver()
    applied = 0
    with driver.session() as session:
        for stmt in SCHEMA_STATEMENTS:
            try:
                session.run(stmt)
                applied += 1
                logger.debug(f"Applied: {stmt[:60]}...")
            except Exception as e:
                logger.warning(f"Schema statement skipped: {e}")
    if applied == 0:
        raise RuntimeError(
            "Failed to apply any Neo4j schema statements. "
            "Check Neo4j availability and credentials."
        )
    logger.success(f"Neo4j schema created/verified ({applied} statements applied)")


def drop_all_data():
    """⚠️  Danger: Wipe all nodes and relationships from the database."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    logger.warning("All data deleted from Neo4j")


def get_schema_info() -> dict:
    """Return a summary of the current schema (node/edge counts)."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        node_counts = {}
        for label in ["Athlete", "Country", "Event", "Games"]:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
            node_counts[label] = result.single()["cnt"]

        rel_counts = {}
        for rel_type in ["PARTICIPATED_IN", "REPRESENTS", "COMPETED_IN", "WON_MEDAL"]:
            result = session.run(
                f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt"
            )
            rel_counts[rel_type] = result.single()["cnt"]

    return {"nodes": node_counts, "relationships": rel_counts}


# ── CLI Entry Point ──────────────────────────────────

if __name__ == "__main__":
    create_schema()
    info = get_schema_info()
    logger.info(f"Schema info: {info}")
