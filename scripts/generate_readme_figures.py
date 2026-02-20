#!/usr/bin/env python3
"""
Generate README figures for Olympus Graph.

The script uses local project data/checkpoints to produce visual assets under:
  docs/figures/
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA
import torch


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_CSV = ROOT / "data" / "raw" / "athlete_events.csv"
DATA_CLEAN_PARQUET = ROOT / "data" / "processed" / "athlete_events_clean.parquet"
EMB_PARQUET = ROOT / "data" / "processed" / "event_embeddings.parquet"
BEST_CKPT = ROOT / "data" / "processed" / "checkpoints" / "best_model.pt"
FINAL_CKPT = ROOT / "data" / "processed" / "checkpoints" / "final_model.pt"
OUT_DIR = ROOT / "docs" / "figures"


def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_events_df() -> pd.DataFrame:
    if DATA_CLEAN_PARQUET.exists():
        df = pd.read_parquet(DATA_CLEAN_PARQUET)
    elif DATA_RAW_CSV.exists():
        df = pd.read_csv(DATA_RAW_CSV)
    else:
        raise FileNotFoundError(
            "No athlete events dataset found at data/processed/athlete_events_clean.parquet "
            "or data/raw/athlete_events.csv."
        )

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "athlete_id" not in df.columns:
        if "id" in df.columns:
            df["athlete_id"] = df["id"].astype(str)
        else:
            df["athlete_id"] = (
                df["name"].astype(str).str.strip()
                + " | "
                + (df["year"].astype(int) - df["age"].fillna(0).astype(int)).astype(str)
            )

    if "event_id" not in df.columns:
        df["event_id"] = df["sport"].astype(str) + " | " + df["event"].astype(str)

    if "games_id" not in df.columns:
        df["games_id"] = df["year"].astype(str) + " " + df["season"].astype(str)

    df["medal"] = df["medal"].replace("nan", np.nan)
    return df


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["figure.titlesize"] = 15
    plt.rcParams["font.size"] = 10
    plt.rcParams["savefig.dpi"] = 170


def _save(fig: plt.Figure, filename: str) -> None:
    out = OUT_DIR / filename
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out.relative_to(ROOT)}")


def make_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.03, 0.37, 0.14, 0.27, "Raw Olympic CSV", "athlete_events.csv"),
        (0.21, 0.37, 0.14, 0.27, "Neo4j Temporal KG", "year-aware edges"),
        (0.39, 0.37, 0.14, 0.27, "Feature Builder", "node tensors + embeddings"),
        (0.57, 0.37, 0.14, 0.27, "Hetero GNN", "GraphSAGE + GATv2"),
        (0.75, 0.37, 0.14, 0.27, "LangGraph Agent", "query + reflection loop"),
    ]
    colors = ["#EAF2FF", "#DFF2F1", "#F3F6DC", "#FCE8D8", "#F5E7FF"]

    for (x, y, w, h, title, subtitle), color in zip(boxes, colors):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.5,
            edgecolor="#3A3A3A",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=9, color="#333333")

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i + 1][0]
        ax.annotate(
            "",
            xy=(x2 - 0.01, 0.505),
            xytext=(x1 + 0.01, 0.505),
            arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#444444"},
        )

    ax.text(0.5, 0.86, "Olympus Graph End-to-End Modeling Pipeline", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.15, "Streamlit dashboard consumes both structured predictions and natural-language explanations.", ha="center", va="center", fontsize=10)

    _save(fig, "pipeline_overview.png")


def make_graph_schema() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.1, 1.1)
    ax.axis("off")

    pos = {
        "Athlete": (0.0, 0.05),
        "Country": (-0.85, 0.65),
        "Event": (0.9, 0.55),
        "Games": (-0.05, -0.8),
    }
    colors = {
        "Athlete": "#FFD166",
        "Country": "#73D2DE",
        "Event": "#A29BFE",
        "Games": "#55D6A6",
    }

    for node, (x, y) in pos.items():
        circle = plt.Circle((x, y), 0.22, color=colors[node], ec="#1F2937", lw=2)
        ax.add_patch(circle)
        ax.text(x, y + 0.03, node, ha="center", va="center", fontsize=12, fontweight="bold")

    def edge(a: str, b: str, label: str, rad: float = 0.0, yoff: float = 0.04) -> None:
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                lw=2,
                color="#374151",
                shrinkA=20,
                shrinkB=20,
                connectionstyle=f"arc3,rad={rad}",
            ),
        )
        lx = (x1 + x2) / 2
        ly = (y1 + y2) / 2 + yoff
        ax.text(lx, ly, label, fontsize=9, fontweight="bold", color="#111827")

    edge("Athlete", "Country", "REPRESENTS", rad=0.1)
    edge("Athlete", "Event", "COMPETED_IN", rad=0.1)
    edge("Athlete", "Event", "WON_MEDAL", rad=-0.2, yoff=-0.08)
    edge("Athlete", "Games", "PARTICIPATED_IN", rad=-0.03, yoff=-0.02)

    ax.text(0, 0.99, "Heterogeneous Graph Schema for Link Prediction", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(
        0,
        -1.02,
        "Reverse edges are added in PyG for message passing on each directed relation.",
        ha="center",
        va="center",
        fontsize=10,
    )

    _save(fig, "graph_schema.png")


def make_temporal_split() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 2.8))
    ax.set_xlim(1894, 2033)
    ax.set_ylim(0, 2)
    ax.set_yticks([])

    ax.barh(1.2, 2015 - 1896 + 1, left=1896, height=0.4, color="#2E86AB", label="Training edges (year < 2016)")
    ax.barh(1.2, 1.0, left=2016, height=0.4, color="#F6AA1C", label="Test ground truth (2016)")
    ax.barh(0.6, 1.0, left=2028, height=0.4, color="#7DCE82", label="Inference targets (2028/2032)")
    ax.barh(0.6, 1.0, left=2032, height=0.4, color="#7DCE82")

    ax.axvline(2016, color="#111827", ls="--", lw=1.3)
    ax.text(2016, 1.67, "Leakage boundary", ha="center", fontsize=9)
    ax.text(1896, 1.75, "Train range", fontsize=9)
    ax.text(2028, 1.75, "Future prediction range", fontsize=9)

    ax.set_xlabel("Olympic year")
    ax.set_title("Temporal Split and Leakage Prevention Strategy")
    ax.legend(loc="upper left", frameon=True)

    _save(fig, "temporal_split.png")


def make_churn_figure(df: pd.DataFrame) -> None:
    season_order = {"Winter": 0, "Summer": 1}
    games = (
        df[["games_id", "year", "season", "athlete_id"]]
        .drop_duplicates()
        .dropna(subset=["games_id", "athlete_id"])
    )
    games_meta = games[["games_id", "year", "season"]].drop_duplicates().copy()
    games_meta["season_rank"] = games_meta["season"].map(season_order).fillna(2)
    games_meta = games_meta.sort_values(["year", "season_rank", "games_id"])

    seen: set[str] = set()
    rows: list[dict[str, float | str | int]] = []
    for _, row in games_meta.iterrows():
        gid = row["games_id"]
        athletes = set(games.loc[games["games_id"] == gid, "athlete_id"].astype(str))
        new_count = len(athletes - seen)
        returning_count = len(athletes & seen)
        total = len(athletes)
        new_share = new_count / total if total else 0.0
        rows.append(
            {
                "games_id": gid,
                "year": int(row["year"]),
                "new_athletes": new_count,
                "returning_athletes": returning_count,
                "new_share": new_share,
            }
        )
        seen |= athletes

    churn = pd.DataFrame(rows)
    x = np.arange(len(churn))

    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.bar(x, churn["new_athletes"], color="#4D96FF", label="New athletes")
    ax.bar(
        x,
        churn["returning_athletes"],
        bottom=churn["new_athletes"],
        color="#75CFB8",
        label="Returning athletes",
    )
    ax.set_ylabel("Unique athletes per Games")
    ax.set_title("Athlete Churn Across Olympic Games")

    every = max(1, len(churn) // 12)
    tick_idx = x[::every]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(churn.iloc[tick_idx]["games_id"], rotation=40, ha="right")

    ax2 = ax.twinx()
    ax2.plot(x, churn["new_share"], color="#111827", lw=2.0, label="New athlete share")
    ax2.set_ylabel("New athlete share")
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    _save(fig, "athlete_churn.png")


def make_label_imbalance(df: pd.DataFrame) -> None:
    edge_cols = ["athlete_id", "event_id", "year"]
    competed = df[edge_cols].drop_duplicates()
    medals = (
        df[df["medal"].notna()][edge_cols + ["medal"]]
        .drop_duplicates(subset=edge_cols)
    )

    positives = len(medals)
    total_competed = len(competed)
    negatives = max(total_competed - positives, 0)
    ratio = negatives / positives if positives else float("inf")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.0))

    axes[0].bar(["Positive\n(WON_MEDAL)", "Negative\n(no medal)"], [positives, negatives], color=["#2E86AB", "#FF6B6B"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Edge count (log scale)")
    axes[0].set_title("Link Prediction Class Imbalance")
    ratio_text = f"Neg:Pos ratio ~ {ratio:.2f}:1" if math.isfinite(ratio) else "Neg:Pos ratio unavailable"
    axes[0].text(0.5, 0.92, ratio_text, transform=axes[0].transAxes, ha="center", fontsize=10)

    medal_counts = medals["medal"].value_counts().reindex(["Gold", "Silver", "Bronze"]).fillna(0)
    axes[1].bar(medal_counts.index, medal_counts.values, color=["#D4AF37", "#BFC5CD", "#CD7F32"])
    axes[1].set_title("Positive Edge Composition by Medal Type")
    axes[1].set_ylabel("Count")

    _save(fig, "label_imbalance.png")


def make_event_embedding_map(df: pd.DataFrame) -> None:
    if not EMB_PARQUET.exists():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "event_embeddings.parquet not found", ha="center", va="center", fontsize=14)
        _save(fig, "event_embedding_map.png")
        return

    emb_df = pd.read_parquet(EMB_PARQUET)
    emb_df = emb_df.dropna(subset=["event_id", "embedding"]).copy()
    if emb_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No event embeddings available", ha="center", va="center", fontsize=14)
        _save(fig, "event_embedding_map.png")
        return

    emb_df["embedding"] = emb_df["embedding"].apply(np.array)
    X = np.vstack(emb_df["embedding"].values)
    if X.shape[0] < 3:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "Not enough embeddings for projection", ha="center", va="center", fontsize=14)
        _save(fig, "event_embedding_map.png")
        return

    event_sport = df[["event_id", "sport"]].drop_duplicates()
    vis = emb_df.merge(event_sport, on="event_id", how="left")
    top_sports = vis["sport"].value_counts().head(8).index
    vis["sport_group"] = np.where(vis["sport"].isin(top_sports), vis["sport"], "Other")

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    vis["pc1"] = coords[:, 0]
    vis["pc2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for sport, grp in vis.groupby("sport_group"):
        ax.scatter(grp["pc1"], grp["pc2"], s=36, alpha=0.72, label=sport)

    ax.set_title("Event Embedding Space (PCA Projection)")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.text(
        0.02,
        0.02,
        f"Explained variance: {(pca.explained_variance_ratio_.sum() * 100):.1f}%",
        transform=ax.transAxes,
        fontsize=9,
    )

    _save(fig, "event_embedding_map.png")


def _load_checkpoint_metrics() -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {
        "best_recall": 0.0,
        "final_recall": 0.0,
        "best_epoch": 0,
        "best_loss": 0.0,
    }
    if BEST_CKPT.exists():
        best = torch.load(BEST_CKPT, map_location="cpu", weights_only=False)
        metrics["best_recall"] = float(best.get("recall_at_k", 0.0))
        metrics["best_epoch"] = int(best.get("epoch", 0))
        metrics["best_loss"] = float(best.get("loss", 0.0))
    if FINAL_CKPT.exists():
        final = torch.load(FINAL_CKPT, map_location="cpu", weights_only=False)
        eval_results = final.get("eval_results", {}) or {}
        metrics["final_recall"] = float(eval_results.get("recall_at_k", 0.0))
    return metrics


def make_model_profile() -> None:
    feature_dims = pd.DataFrame(
        {
            "node_type": ["event", "athlete", "country", "games"],
            "feature_dim": [384, 9, 3, 2],
        }
    )
    metrics = _load_checkpoint_metrics()

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.1))

    axes[0].barh(feature_dims["node_type"], feature_dims["feature_dim"], color="#5865F2")
    axes[0].invert_yaxis()
    axes[0].set_title("Input Feature Dimensionality by Node Type")
    axes[0].set_xlabel("Feature dimensions")

    recall_vals = [metrics["best_recall"], metrics["final_recall"]]
    axes[1].bar(["Best Recall@3", "Final Recall@3"], recall_vals, color=["#0EA5E9", "#22C55E"])
    axes[1].set_ylim(0, max(0.2, max(recall_vals) * 1.35))
    axes[1].set_title("Checkpointed Evaluation Snapshot")
    axes[1].set_ylabel("Recall@3")
    axes[1].text(
        0.04,
        0.90,
        f"Best epoch: {metrics['best_epoch']}\nBest BCE loss: {metrics['best_loss']:.4f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#F3F4F6", ec="#D1D5DB"),
    )

    _save(fig, "model_profile.png")


def main() -> None:
    _ensure_out_dir()
    _style()

    df = _load_events_df()

    make_pipeline_overview()
    make_graph_schema()
    make_temporal_split()
    make_churn_figure(df)
    make_label_imbalance(df)
    make_event_embedding_map(df)
    make_model_profile()

    print(f"all figures written to: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"figure generation failed: {exc}", file=sys.stderr)
        raise
