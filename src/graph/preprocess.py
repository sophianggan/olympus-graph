"""
Olympus Graph – Data Preprocessing
Compute sentence embeddings for Events and enrich Country nodes.

Steps:
1. Compute 384-dim embeddings for each Event using 'all-MiniLM-L6-v2'
2. Store embeddings as properties on Event nodes in Neo4j
3. Enrich Country nodes with GDP / population / host-flag data
4. Save processed feature matrices as parquet for GNN consumption
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
)
from src.utils import get_neo4j_driver, run_cypher, timed


# ── Host City → Country Mapping (known Olympic hosts) ─

HOST_COUNTRY_MAP = {
    # Summer Olympics hosts (year → NOC)
    1896: "GRE", 1900: "FRA", 1904: "USA", 1908: "GBR", 1912: "SWE",
    1920: "BEL", 1924: "FRA", 1928: "NED", 1932: "USA", 1936: "GER",
    1948: "GBR", 1952: "FIN", 1956: "AUS", 1960: "ITA", 1964: "JPN",
    1968: "MEX", 1972: "GER", 1976: "CAN", 1980: "URS", 1984: "USA",
    1988: "KOR", 1992: "ESP", 1996: "USA", 2000: "AUS", 2004: "GRE",
    2008: "CHN", 2012: "GBR", 2016: "BRA", 2020: "JPN", 2024: "FRA",
    2028: "USA",  # Future: LA 2028
    # Winter Olympics hosts
    # (simplified, add as needed)
}

# ── Synthetic GDP Data (per-capita, thousands USD, approximate) ─

GDP_PER_CAPITA = {
    "USA": 76000, "CHN": 12500, "JPN": 34000, "GER": 51000, "GBR": 46000,
    "FRA": 44000, "AUS": 52000, "CAN": 52000, "KOR": 33000, "ITA": 35000,
    "RUS": 12000, "BRA": 9000, "NED": 57000, "ESP": 30000, "SWE": 56000,
    "NOR": 82000, "KEN": 2000, "JAM": 5500, "ETH": 1200, "CUB": 10000,
    "NZL": 42000, "MEX": 10000, "IND": 2400, "RSA": 6000, "GRE": 20000,
    "TUR": 10000, "POL": 18000, "UKR": 4000, "EGY": 4000, "ARG": 10000,
    "COL": 6500, "NGR": 2200, "ROM": 15000, "HUN": 18000, "CZE": 27000,
    "BEL": 50000, "AUT": 53000, "SUI": 93000, "DEN": 68000, "FIN": 54000,
    "POR": 25000, "IRL": 100000, "ISR": 53000, "THA": 7500, "PHI": 3700,
    "MAR": 3800, "TUN": 3800, "ALG": 3900, "PAK": 1500, "IRI": 6000,
}

POPULATION = {
    "USA": 331e6, "CHN": 1412e6, "JPN": 125e6, "GER": 83e6, "GBR": 67e6,
    "FRA": 67e6, "AUS": 26e6, "CAN": 38e6, "KOR": 52e6, "ITA": 59e6,
    "RUS": 144e6, "BRA": 214e6, "NED": 17e6, "ESP": 47e6, "SWE": 10e6,
    "NOR": 5.4e6, "KEN": 54e6, "JAM": 3e6, "ETH": 120e6, "CUB": 11e6,
}


# ── Embedding Computation ────────────────────────────

@timed
def compute_event_embeddings():
    """
    Compute sentence embeddings for all Event nodes and store them in Neo4j.
    Uses 'all-MiniLM-L6-v2' (384-dim).
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Fetch all events
    events = run_cypher("MATCH (e:Event) RETURN e.event_id AS event_id, e.event AS event")
    logger.info(f"Computing embeddings for {len(events)} events")

    # Build descriptions for embedding
    texts = [ev["event"] for ev in events]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)

    # Store embeddings back in Neo4j
    driver = get_neo4j_driver()
    with driver.session() as session:
        for ev, emb in zip(events, embeddings):
            emb_list = emb.tolist()
            session.run(
                """
                MATCH (e:Event {event_id: $event_id})
                SET e.embedding = $embedding
                """,
                {"event_id": ev["event_id"], "embedding": emb_list},
            )

    # Also save as parquet for direct GNN consumption
    emb_df = pd.DataFrame({
        "event_id": [ev["event_id"] for ev in events],
        "embedding": [emb.tolist() for emb in embeddings],
    })
    emb_path = DATA_PROCESSED_DIR / "event_embeddings.parquet"
    emb_df.to_parquet(emb_path, index=False)
    logger.info(f"Saved event embeddings to {emb_path}")

    return embeddings


# ── Country Enrichment ───────────────────────────────

@timed
def enrich_country_nodes():
    """
    Add GDP, population, and host-flag data to Country nodes.
    Since these are dynamic by year, we store them as properties
    and the snapshot logic handles temporal filtering.
    """
    driver = get_neo4j_driver()

    # Set GDP & Population
    with driver.session() as session:
        for noc, gdp in GDP_PER_CAPITA.items():
            pop = POPULATION.get(noc, 10e6)  # Default population
            session.run(
                """
                MATCH (c:Country {noc: $noc})
                SET c.gdp = $gdp, c.population = $population
                """,
                {"noc": noc, "gdp": gdp, "population": pop},
            )

    logger.info(f"Enriched {len(GDP_PER_CAPITA)} countries with GDP data")

    # Set host flags on Games nodes
    with driver.session() as session:
        for year, host_noc in HOST_COUNTRY_MAP.items():
            session.run(
                """
                MATCH (g:Games)
                WHERE g.year = $year
                SET g.host_noc = $host_noc
                """,
                {"year": year, "host_noc": host_noc},
            )

    logger.info(f"Set host flags for {len(HOST_COUNTRY_MAP)} Games")


# ── Feature Matrix Export ────────────────────────────

@timed
def export_feature_matrices(max_year: int = 2024):
    """
    Export pre-computed feature matrices for the GNN.
    All features are computed ONLY from data before max_year.
    """
    # Athlete features
    athlete_query = """
    MATCH (a:Athlete)-[p:PARTICIPATED_IN]->(g:Games)
    WHERE g.year < $max_year
    WITH a,
         avg(p.performance) AS avg_performance,
         max(p.performance) AS max_performance,
         count(p)           AS num_participations,
         max(g.year)        AS last_year,
         collect(DISTINCT g.year) AS years
    MATCH (a)-[:REPRESENTS]->(c:Country)
    RETURN
        a.athlete_id       AS athlete_id,
        a.sex              AS sex,
        a.birth_year       AS birth_year,
        a.height           AS height,
        a.weight           AS weight,
        avg_performance,
        max_performance,
        num_participations,
        last_year,
        size(years)        AS num_games,
        c.noc              AS noc,
        c.gdp              AS gdp,
        c.population       AS population
    """
    athletes_df = pd.DataFrame(run_cypher(athlete_query, {"max_year": max_year}))

    # Compute normalized features
    if len(athletes_df) > 0:
        athletes_df["age_at_last"] = athletes_df["last_year"] - athletes_df["birth_year"]
        for col in ["avg_performance", "height", "weight", "age_at_last", "num_games"]:
            if col in athletes_df.columns:
                col_min = athletes_df[col].min()
                col_max = athletes_df[col].max()
                if col_max > col_min:
                    athletes_df[f"{col}_norm"] = (
                        (athletes_df[col] - col_min) / (col_max - col_min)
                    )
                else:
                    athletes_df[f"{col}_norm"] = 0.0

    athlete_path = DATA_PROCESSED_DIR / f"athlete_features_{max_year}.parquet"
    athletes_df.to_parquet(athlete_path, index=False)
    logger.info(f"Saved athlete features to {athlete_path} ({len(athletes_df)} athletes)")

    return athletes_df


# ── Master Preprocessing Pipeline ────────────────────

@timed
def run_preprocessing():
    """Full preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("OLYMPUS GRAPH — Preprocessing Pipeline")
    logger.info("=" * 60)

    compute_event_embeddings()
    enrich_country_nodes()
    export_feature_matrices(max_year=2024)  # For training
    export_feature_matrices(max_year=2028)  # For future predictions

    logger.success("Preprocessing complete!")


if __name__ == "__main__":
    run_preprocessing()
