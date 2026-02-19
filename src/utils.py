"""
Olympus Graph – Shared Utilities
"""

from __future__ import annotations

import functools
import time
from typing import Any

from loguru import logger
from neo4j import GraphDatabase

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


# ── Neo4j Driver Singleton ────────────────────────────

_driver = None


def get_neo4j_driver():
    """Return a singleton Neo4j driver instance."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        # Fail fast so the caller gets a clear startup error instead of a later query crash.
        _driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {NEO4J_URI}")
    return _driver


def close_neo4j_driver():
    """Close the Neo4j driver."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
        logger.info("Neo4j driver closed")


def run_cypher(query: str, parameters: dict | None = None) -> list[dict[str, Any]]:
    """Execute a Cypher query and return all records as dicts."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]


# ── Timing Decorator ─────────────────────────────────

def timed(func):
    """Log execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# ── Batch Helpers ────────────────────────────────────

def chunked(iterable, size: int):
    """Yield successive chunks of `size` from an iterable."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
