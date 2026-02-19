"""
Olympus Graph – Data Ingestion Pipeline
Loads Kaggle CSVs into Neo4j with temporal-aware schema.

Expected files in data/raw/:
  - athlete_events.csv  (the main Kaggle Olympics dataset)
  - noc_regions.csv     (NOC → region mapping, optional)

CRITICAL: All relationships carry a `year` property so the graph
supports temporal "snapshots" — the GNN never sees future data.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR
from src.utils import get_neo4j_driver, chunked, timed
from src.graph.schema import create_schema


# ── Constants ─────────────────────────────────────────

BATCH_SIZE = 5000
MEDAL_MAP = {"Gold": "Gold", "Silver": "Silver", "Bronze": "Bronze"}


# ── Data Loading & Cleaning ──────────────────────────

def ensure_raw_data_exists() -> Path:
    """Return the athlete_events.csv path if it exists, otherwise raise with guidance."""
    csv_path = DATA_RAW_DIR / "athlete_events.csv"
    if csv_path.exists():
        return csv_path

    raise FileNotFoundError(
        f"Missing required dataset: {csv_path}\n"
        "Use one of the following:\n"
        "1) Download Kaggle data to data/raw/athlete_events.csv\n"
        "2) Run sample bootstrap data: python -m src.graph.bootstrap_data"
    )


@timed
def load_and_clean_data() -> pd.DataFrame:
    """Load athlete_events.csv, clean, and return a DataFrame."""
    csv_path = ensure_raw_data_exists()

    logger.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Clean core fields
    df["name"] = df["name"].str.strip()
    df["sex"] = df["sex"].str.strip()
    df["noc"] = df["noc"].str.strip()
    df["sport"] = df["sport"].str.strip()
    df["event"] = df["event"].str.strip()
    df["season"] = df["season"].str.strip()

    # Parse year
    df["year"] = df["year"].astype(int)

    # Create composite IDs
    # Athlete ID: Name + Age approximation (year - age -> birth year)
    df["birth_year"] = df["year"] - df["age"].fillna(0).astype(int)
    df["athlete_id"] = df["name"] + " | " + df["birth_year"].astype(str)

    # Event ID: Sport + Event Name
    df["event_id"] = df["sport"] + " | " + df["event"]

    # Games ID: Year + Season
    df["games_id"] = df["year"].astype(str) + " " + df["season"]

    # Normalize performance: use medal as a proxy (Gold=3, Silver=2, Bronze=1, NA=0)
    df["performance"] = df["medal"].map({"Gold": 3, "Silver": 2, "Bronze": 1}).fillna(0)

    # Fill missing values
    df["age"] = df["age"].fillna(df["age"].median())
    df["height"] = df["height"].fillna(df["height"].median())
    df["weight"] = df["weight"].fillna(df["weight"].median())

    # Save processed version
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = DATA_PROCESSED_DIR / "athlete_events_clean.parquet"
    df.to_parquet(processed_path, index=False)
    logger.info(f"Saved cleaned data to {processed_path} ({len(df)} rows)")

    return df


# ── Node Creation ────────────────────────────────────

@timed
def create_athlete_nodes(df: pd.DataFrame):
    """Create Athlete nodes from unique athletes."""
    athletes = (
        df.groupby("athlete_id")
        .agg({
            "name": "first",
            "sex": "first",
            "birth_year": "first",
            "height": "median",
            "weight": "median",
        })
        .reset_index()
    )

    driver = get_neo4j_driver()
    query = """
    UNWIND $batch AS row
    MERGE (a:Athlete {athlete_id: row.athlete_id})
    ON CREATE SET
        a.name       = row.name,
        a.sex        = row.sex,
        a.birth_year = row.birth_year,
        a.height     = row.height,
        a.weight     = row.weight
    """

    with driver.session() as session:
        for chunk in tqdm(
            list(chunked(athletes.to_dict("records"), BATCH_SIZE)),
            desc="Athletes",
        ):
            session.run(query, {"batch": chunk})

    logger.info(f"Created {len(athletes)} Athlete nodes")


@timed
def create_country_nodes(df: pd.DataFrame):
    """Create Country nodes from unique NOCs."""
    countries = df[["noc"]].drop_duplicates()

    driver = get_neo4j_driver()
    query = """
    UNWIND $batch AS row
    MERGE (c:Country {noc: row.noc})
    """

    with driver.session() as session:
        for chunk in chunked(countries.to_dict("records"), BATCH_SIZE):
            session.run(query, {"batch": chunk})

    logger.info(f"Created {len(countries)} Country nodes")


@timed
def create_event_nodes(df: pd.DataFrame):
    """Create Event nodes from unique events."""
    events = (
        df.groupby("event_id")
        .agg({"sport": "first", "event": "first"})
        .reset_index()
    )

    driver = get_neo4j_driver()
    query = """
    UNWIND $batch AS row
    MERGE (e:Event {event_id: row.event_id})
    ON CREATE SET
        e.sport = row.sport,
        e.event = row.event
    """

    with driver.session() as session:
        for chunk in chunked(events.to_dict("records"), BATCH_SIZE):
            session.run(query, {"batch": chunk})

    logger.info(f"Created {len(events)} Event nodes")


@timed
def create_games_nodes(df: pd.DataFrame):
    """Create Games nodes from unique games."""
    games = (
        df.groupby("games_id")
        .agg({"year": "first", "season": "first", "city": "first"})
        .reset_index()
    )

    driver = get_neo4j_driver()
    query = """
    UNWIND $batch AS row
    MERGE (g:Games {games_id: row.games_id})
    ON CREATE SET
        g.year   = row.year,
        g.season = row.season,
        g.city   = row.city
    """

    with driver.session() as session:
        for chunk in chunked(games.to_dict("records"), BATCH_SIZE):
            session.run(query, {"batch": chunk})

    logger.info(f"Created {len(games)} Games nodes")


# ── Edge Creation (Temporal) ─────────────────────────

@timed
def create_participated_in_edges(df: pd.DataFrame):
    """Create PARTICIPATED_IN edges: Athlete → Games (with year, age, performance)."""
    participations = (
        df.groupby(["athlete_id", "games_id"])
        .agg({"age": "first", "performance": "max", "year": "first"})
        .reset_index()
    )

    driver = get_neo4j_driver()
    query = """
    UNWIND $batch AS row
    MATCH (a:Athlete {athlete_id: row.athlete_id})
    MATCH (g:Games {games_id: row.games_id})
    MERGE (a)-[r:PARTICIPATED_IN {year: row.year}]->(g)
    ON CREATE SET
        r.age         = toInteger(row.age),
        r.performance = toFloat(row.performance)
    """

    with driver.session() as session:
        for chunk in tqdm(
            list(chunked(participations.to_dict("records"), BATCH_SIZE)),
            desc="PARTICIPATED_IN",
        ):
            session.run(query, {"batch": chunk})

    logger.info(f"Created {len(participations)} PARTICIPATED_IN edges")


@timed
def create_represents_edges(df: pd.DataFrame):
    """Create REPRESENTS edges: Athlete → Country."""
    reps = df[["athlete_id", "noc"]].drop_duplicates()

    driver = get_neo4j_driver()
    query = """
    UNWIND $batch AS row
    MATCH (a:Athlete {athlete_id: row.athlete_id})
    MATCH (c:Country {noc: row.noc})
    MERGE (a)-[:REPRESENTS]->(c)
    """

    with driver.session() as session:
        for chunk in tqdm(
            list(chunked(reps.to_dict("records"), BATCH_SIZE)),
            desc="REPRESENTS",
        ):
            session.run(query, {"batch": chunk})

    logger.info(f"Created {len(reps)} REPRESENTS edges")


@timed
def create_competed_in_edges(df: pd.DataFrame):
    """Create COMPETED_IN edges: Athlete → Event (with year for temporal filtering)."""
    competitions = (
        df.groupby(["athlete_id", "event_id", "year"])
        .size()
        .reset_index(name="count")
    )

    driver = get_neo4j_driver()
    query = """
    UNWIND $batch AS row
    MATCH (a:Athlete {athlete_id: row.athlete_id})
    MATCH (e:Event {event_id: row.event_id})
    MERGE (a)-[r:COMPETED_IN {year: row.year}]->(e)
    """

    with driver.session() as session:
        for chunk in tqdm(
            list(chunked(competitions.to_dict("records"), BATCH_SIZE)),
            desc="COMPETED_IN",
        ):
            session.run(query, {"batch": chunk})

    logger.info(f"Created {len(competitions)} COMPETED_IN edges")


@timed
def create_won_medal_edges(df: pd.DataFrame):
    """Create WON_MEDAL edges: Athlete → Event (only for medal winners, with year + medal type)."""
    medals = df[df["medal"].notna()].copy()
    medals = (
        medals.groupby(["athlete_id", "event_id", "year"])
        .agg({"medal": "first"})
        .reset_index()
    )

    driver = get_neo4j_driver()
    query = """
    UNWIND $batch AS row
    MATCH (a:Athlete {athlete_id: row.athlete_id})
    MATCH (e:Event {event_id: row.event_id})
    MERGE (a)-[r:WON_MEDAL {year: row.year}]->(e)
    ON CREATE SET r.medal = row.medal
    """

    with driver.session() as session:
        for chunk in tqdm(
            list(chunked(medals.to_dict("records"), BATCH_SIZE)),
            desc="WON_MEDAL",
        ):
            session.run(query, {"batch": chunk})

    logger.info(f"Created {len(medals)} WON_MEDAL edges")


# ── Master Ingestion Pipeline ────────────────────────

@timed
def run_ingestion():
    """Full ingestion pipeline: schema → nodes → edges."""
    logger.info("=" * 60)
    logger.info("OLYMPUS GRAPH — Data Ingestion Pipeline")
    logger.info("=" * 60)

    # Step 0: Validate local data early for a faster failure path.
    ensure_raw_data_exists()

    # Step 1: Create schema
    create_schema()

    # Step 2: Load and clean data
    df = load_and_clean_data()

    # Step 3: Create nodes
    create_athlete_nodes(df)
    create_country_nodes(df)
    create_event_nodes(df)
    create_games_nodes(df)

    # Step 4: Create edges (temporal)
    create_participated_in_edges(df)
    create_represents_edges(df)
    create_competed_in_edges(df)
    create_won_medal_edges(df)

    logger.success("Ingestion complete!")


# ── CLI Entry Point ──────────────────────────────────

if __name__ == "__main__":
    run_ingestion()
