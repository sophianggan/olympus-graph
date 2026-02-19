"""
Olympus Graph – Streamlit Dashboard
Interactive UI for querying the Olympus Graph agent.

Features:
  - Natural language input
  - Prediction table (Top-K athletes with probabilities)
  - LLM-generated explanation
  - Interactive graph visualization (pyvis)
"""

from __future__ import annotations

import json
import streamlit as st
from pyvis.network import Network
import tempfile
import os
import sys
from pathlib import Path

# Must be first Streamlit command
st.set_page_config(
    page_title="🏛️ Olympus Graph",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Imports (after page config) ──────────────────────

# Ensure the project root is importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.workflow import ask
from src.agent.tools import graph_query_tool, model_predict_tool
from src.graph.snapshot import get_athlete_neighborhood
from src.utils import run_cypher


# ══════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/5c/Olympic_rings_without_rims.svg", width=200)
    st.title("🏛️ Olympus Graph")
    st.markdown(
        "**Neuro-Symbolic AI** for Olympic Medal Prediction\n\n"
        "Powered by:\n"
        "- 📊 Neo4j Knowledge Graph\n"
        "- 🧠 Graph Neural Networks (PyG)\n"
        "- 🤖 LangGraph Agent\n"
    )

    st.divider()

    st.subheader("⚙️ Settings")
    target_year = st.selectbox("Prediction Year", [2028, 2032], index=0)
    top_k = st.slider("Top-K Predictions", 1, 10, 3)

    st.divider()

    st.subheader("📋 Example Queries")
    examples = [
        "Who will win the Men's 100m in 2028?",
        "Who won the most Gold medals in Swimming?",
        "Show me Usain Bolt's Olympic history",
        "Which country has the most medals in Athletics?",
        "Predict the Women's Marathon winner for 2028",
        "Who won Gold in Gymnastics at the 2020 Olympics?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["query_input"] = ex


# ══════════════════════════════════════════════════════
# Main Content
# ══════════════════════════════════════════════════════

st.title("🏅 Olympus Graph — Olympic Medal Predictor")
st.markdown(
    "Ask me anything about Olympic history or future predictions. "
    "I use a **Knowledge Graph + GNN** to answer."
)

# Query input
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "Ask a question:",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g., Who will win the Men's 100m in 2028?",
        key="main_query",
    )
with col2:
    st.write("")  # Spacing
    st.write("")
    ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════
# Process Query
# ══════════════════════════════════════════════════════

if ask_button and query:
    with st.spinner("🏛️ Olympus is thinking..."):

        # ── Get Agent Answer ──────────────────────
        try:
            answer = ask(query)
        except Exception as e:
            answer = f"⚠️ Agent error: {str(e)}"

        # ── Display Answer ────────────────────────
        st.divider()

        # Check if it's a prediction query
        is_prediction = any(
            kw in query.lower()
            for kw in ["will", "predict", "2028", "2032", "future", "next"]
        )

        if is_prediction:
            st.subheader("🔮 Prediction Results")

            # Run the model prediction tool directly for structured output
            try:
                # Extract event name heuristically
                event_name = query.replace("?", "").strip()
                for remove in ["who will win the", "who will win", "predict the",
                              "predict", "winner for", "in 2028", "in 2032",
                              "winner of", "winner"]:
                    event_name = event_name.lower().replace(remove, "").strip()

                pred_result = model_predict_tool(
                    event_name=event_name,
                    target_year=target_year,
                    top_k=top_k,
                )

                if pred_result["success"] and pred_result["predictions"]:
                    # Prediction Table
                    st.markdown(f"**Event:** {pred_result['event']}")
                    st.markdown(f"**Target Year:** {pred_result['target_year']}")
                    if pred_result.get("host_country"):
                        st.markdown(f"**Host Country:** {pred_result['host_country']}")

                    # Medal prediction table
                    medal_icons = {0: "🥇", 1: "🥈", 2: "🥉"}
                    table_data = []
                    for i, p in enumerate(pred_result["predictions"]):
                        table_data.append({
                            "": medal_icons.get(i, f"#{i+1}"),
                            "Athlete": p["name"],
                            "Country": p["country"],
                            "Age": p.get("age_at_games", "N/A"),
                            "Probability": f"{p['probability']:.1%}",
                        })

                    st.table(table_data)

                    # ── Graph Visualization ───────────
                    st.subheader("🕸️ Athlete Network")
                    top_athlete_id = pred_result["predictions"][0]["athlete_id"]

                    try:
                        neighborhood = get_athlete_neighborhood(
                            top_athlete_id, max_year=target_year
                        )
                        if neighborhood:
                            net = Network(
                                height="500px",
                                width="100%",
                                bgcolor="#0e1117",
                                font_color="white",
                                directed=True,
                            )
                            net.barnes_hut()

                            # Add central athlete
                            athlete_info = neighborhood.get("athlete", {})
                            net.add_node(
                                "athlete_center",
                                label=athlete_info.get("name", "Athlete"),
                                color="#FFD700",
                                size=30,
                                title=f"Athlete: {athlete_info.get('name', '')}",
                            )

                            # Add countries
                            for c in neighborhood.get("countries", []):
                                if c.get("noc"):
                                    net.add_node(
                                        f"country_{c['noc']}",
                                        label=c["noc"],
                                        color="#4CAF50",
                                        size=20,
                                        title=f"Country: {c['noc']}",
                                    )
                                    net.add_edge("athlete_center", f"country_{c['noc']}", label="REPRESENTS")

                            # Add events (limit to 10)
                            for i, e in enumerate(neighborhood.get("events", [])[:10]):
                                if e.get("event_id"):
                                    net.add_node(
                                        f"event_{i}",
                                        label=e.get("event", e["event_id"])[:30],
                                        color="#2196F3",
                                        size=15,
                                        title=f"Event: {e.get('event', '')}",
                                    )
                                    net.add_edge("athlete_center", f"event_{i}", label="COMPETED_IN")

                            # Add medals
                            for i, m in enumerate(neighborhood.get("medals", [])):
                                medal_color = {
                                    "Gold": "#FFD700",
                                    "Silver": "#C0C0C0",
                                    "Bronze": "#CD7F32",
                                }.get(m.get("medal", ""), "#999999")

                                if m.get("event"):
                                    net.add_node(
                                        f"medal_{i}",
                                        label=f"🏅 {m.get('medal', '')}",
                                        color=medal_color,
                                        size=12,
                                        title=f"{m.get('medal', '')} - {m.get('event', '')} ({m.get('year', '')})",
                                    )
                                    net.add_edge("athlete_center", f"medal_{i}", label="WON_MEDAL")

                            # Add games (limit to 8)
                            for i, g in enumerate(neighborhood.get("games", [])[:8]):
                                if g.get("games_id"):
                                    net.add_node(
                                        f"games_{i}",
                                        label=str(g.get("year", "")),
                                        color="#FF9800",
                                        size=12,
                                        title=f"Games: {g.get('games_id', '')} ({g.get('city', '')})",
                                    )
                                    net.add_edge("athlete_center", f"games_{i}", label="PARTICIPATED_IN")

                            # Render
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".html", mode="w"
                            ) as f:
                                net.save_graph(f.name)
                                with open(f.name, "r") as html_file:
                                    html_content = html_file.read()
                                st.components.v1.html(html_content, height=520, scrolling=True)
                                os.unlink(f.name)

                    except Exception as viz_err:
                        st.warning(f"Graph visualization unavailable: {viz_err}")

            except Exception as pred_err:
                st.warning(f"Direct prediction unavailable: {pred_err}")

        # ── LLM Explanation ───────────────────────
        st.subheader("💬 AI Analysis")
        st.markdown(answer)

    # Clear query input
    if "query_input" in st.session_state:
        del st.session_state["query_input"]


# ══════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("📊 Data: Kaggle 120 Years of Olympic History")
with col2:
    st.caption("🧠 Model: Heterogeneous GraphSAGE + GATv2")
with col3:
    st.caption("🤖 Agent: LangGraph Self-Correcting Workflow")
