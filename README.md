# 🏛️ Olympus Graph

**A Neuro-Symbolic AI Agent that predicts Olympic medalists using Graph Neural Networks (GNNs) and explains its reasoning via an LLM agent.**

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐
│  Kaggle CSV  │────▶│  Neo4j Graph  │────▶│  PyG GNN     │────▶│ LangGraph │
│  (Raw Data)  │     │  (Temporal)   │     │  (Link Pred)  │     │  (Agent)  │
└─────────────┘     └──────────────┘     └──────────────┘     └─────┬─────┘
                                                                     │
                                                              ┌──────▼──────┐
                                                              │  Streamlit  │
                                                              │  Dashboard  │
                                                              └─────────────┘
```

## Project Structure

```
/olympus-graph
├── data/
│   ├── raw/              # Place Kaggle CSVs here
│   └── processed/        # Intermediate parquet files
├── src/
│   ├── config.py         # Central configuration
│   ├── graph/            # Neo4j ingestion & schema logic
│   │   ├── schema.py     # Constraints & indexes
│   │   ├── ingest.py     # CSV → Neo4j (temporal)
│   │   └── snapshot.py   # Time-travel graph queries
│   ├── model/            # GNN architecture (PyG)
│   │   ├── gnn.py        # HeteroGraphSAGE / GATv2
│   │   ├── dataset.py    # Neo4j → PyG HeteroData
│   │   ├── train.py      # Training loop
│   │   └── evaluate.py   # Recall@K metrics
│   ├── agent/            # LangGraph workflows
│   │   ├── tools.py      # GraphQueryTool, ModelPredictTool
│   │   └── workflow.py   # Parser→Generator→Executor→Reflector→Answer
│   └── app/              # Streamlit dashboard
│       └── dashboard.py  # UI with predictions + graph viz
├── notebooks/            # EDA only
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Prerequisites
- Python 3.11 recommended (3.10+ supported)
- Neo4j running locally (Docker) or AuraDB connection
- OpenAI API key (for LangGraph agent)

### 2. Installation
```bash
cd olympus-graph
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configuration
```bash
cp .env.example .env
# Edit .env with your Neo4j and Groq credentials
```

Default LLM provider is Groq (`LLM_PROVIDER=groq`).

### 4. Start Neo4j (Docker)
```bash
docker run -d --name olympus-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.26-community
```

### 5. Data Setup
Place the Kaggle Olympics dataset CSVs in `data/raw/`:
- `athlete_events.csv` (120,000+ rows of Olympic history)
- Optionally: `noc_regions.csv`, `gdp_data.csv`

If you want a quick local smoke test without Kaggle download:
```bash
python -m src.graph.bootstrap_data
```

### 6. Run Pipeline
```bash
# Phase 1: Ingest data into Neo4j
python -m src.graph.ingest

# Phase 1.5: Preprocess features
python -m src.graph.preprocess

# Phase 2: Train GNN
python -m src.model.train

# Phase 4: Launch dashboard
streamlit run src/app/dashboard.py
```

## Implementation Phases

| Phase | Component | Description |
|-------|-----------|-------------|
| 1 | Temporal Knowledge Graph | Neo4j schema with time-travel snapshots |
| 2 | Link Prediction GNN | Heterogeneous GraphSAGE predicting `WON_MEDAL` edges |
| 3 | Self-Correcting Agent | LangGraph: Parser → Generator → Executor → Reflector → Answer |
| 4 | Streamlit Interface | Predictions + Explanations + Graph Visualization |

## Key Design Decisions

- **Temporal Snapshots**: The graph supports "time-travel" — when predicting for a holdout year, the GNN never sees that year's edges.
- **Heterogeneous Graph**: 4 node types (Athlete, Country, Event, Games) with typed edges.
- **Event Embeddings**: `all-MiniLM-L6-v2` captures semantic similarity (e.g., "100m Sprint" ≈ "200m Sprint").
- **Self-Correcting Agent**: LangGraph's Reflector node catches errors and rewrites queries automatically.

## Evaluation
- **Metric**: Recall@3 — Did the actual Gold medalist appear in the model's top-3 predictions?
- **Train**: Games 1896–2015
- **Test**: Games 2016 (Hold-out)
