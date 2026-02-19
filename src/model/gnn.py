"""
Olympus Graph – GNN Model Architecture
Heterogeneous Graph Neural Network for Olympic Medal Link Prediction.

Architecture: HeteroGraphSAGE with GATv2 attention for the final layer.

Node Types: Athlete, Country, Event, Games
Edge Types: PARTICIPATED_IN, REPRESENTS, COMPETED_IN, WON_MEDAL

Task: Link Prediction — predict probability of (Athlete)-[WON_MEDAL]->(Event)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    SAGEConv,
    GATv2Conv,
    HeteroConv,
    Linear,
    to_hetero,
)
from torch_geometric.data import HeteroData

from src.config import GNN_HIDDEN_DIM, GNN_NUM_LAYERS, GNN_DROPOUT, EMBEDDING_DIM


# ── Node Feature Encoders ────────────────────────────

class AthleteEncoder(nn.Module):
    """
    Encode athlete features into a fixed-dim vector.
    Input features: [avg_performance_norm, max_performance_norm, age_norm,
                     height_norm, weight_norm, num_games_norm, sex_binary]
    """

    def __init__(self, in_dim: int = 7, hidden_dim: int = GNN_HIDDEN_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(GNN_DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class CountryEncoder(nn.Module):
    """
    Encode country features: [gdp_norm, population_norm, is_host]
    """

    def __init__(self, in_dim: int = 3, hidden_dim: int = GNN_HIDDEN_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class EventEncoder(nn.Module):
    """
    Encode event features: 384-dim sentence embedding from 'all-MiniLM-L6-v2'.
    """

    def __init__(self, in_dim: int = EMBEDDING_DIM, hidden_dim: int = GNN_HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class GamesEncoder(nn.Module):
    """
    Encode Games features: [year_norm, season_binary]
    """

    def __init__(self, in_dim: int = 2, hidden_dim: int = GNN_HIDDEN_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ── Heterogeneous GNN ────────────────────────────────

class OlympusHeteroGNN(nn.Module):
    """
    Heterogeneous GNN using GraphSAGE message-passing layers
    with a final GATv2 attention layer.

    Supports 4 node types and 4+ edge types.
    Predicts link probability for (Athlete)-[WON_MEDAL]->(Event).
    """

    def __init__(
        self,
        hidden_dim: int = GNN_HIDDEN_DIM,
        num_layers: int = GNN_NUM_LAYERS,
        dropout: float = GNN_DROPOUT,
        metadata: tuple | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Node-type-specific encoders
        self.athlete_encoder = AthleteEncoder(hidden_dim=hidden_dim)
        self.country_encoder = CountryEncoder(hidden_dim=hidden_dim)
        self.event_encoder = EventEncoder(hidden_dim=hidden_dim)
        self.games_encoder = GamesEncoder(hidden_dim=hidden_dim)

        # Heterogeneous message-passing layers
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            # Use GATv2 for the last layer, SAGEConv for others
            if i < num_layers - 1:
                conv_dict = {}
                if metadata is not None:
                    for edge_type in metadata[1]:
                        conv_dict[edge_type] = SAGEConv(
                            (hidden_dim, hidden_dim), hidden_dim
                        )
                self.convs.append(HeteroConv(conv_dict, aggr="mean"))
            else:
                conv_dict = {}
                if metadata is not None:
                    for edge_type in metadata[1]:
                        conv_dict[edge_type] = GATv2Conv(
                            (hidden_dim, hidden_dim),
                            hidden_dim // 4,
                            heads=4,
                            concat=True,
                            dropout=dropout,
                            add_self_loops=False,
                        )
                self.convs.append(HeteroConv(conv_dict, aggr="mean"))

        # Link prediction decoder: inner product + MLP
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_nodes(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Encode raw features into initial embeddings per node type."""
        x_dict = {}

        if "athlete" in data.node_types and data["athlete"].x is not None:
            x_dict["athlete"] = self.athlete_encoder(data["athlete"].x)
        if "country" in data.node_types and data["country"].x is not None:
            x_dict["country"] = self.country_encoder(data["country"].x)
        if "event" in data.node_types and data["event"].x is not None:
            x_dict["event"] = self.event_encoder(data["event"].x)
        if "games" in data.node_types and data["games"].x is not None:
            x_dict["games"] = self.games_encoder(data["games"].x)

        return x_dict

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """
        Forward pass: encode nodes → message passing → return node embeddings.
        """
        x_dict = self.encode_nodes(data)

        # Message passing
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, data.edge_index_dict)
            if i < self.num_layers - 1:
                x_dict = {
                    key: F.relu(F.dropout(x, p=self.dropout, training=self.training))
                    for key, x in x_dict.items()
                }

        return x_dict

    def predict_link(
        self,
        athlete_emb: torch.Tensor,
        event_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict probability of WON_MEDAL link between athlete(s) and event(s).

        Args:
            athlete_emb: (N, hidden_dim) athlete embeddings
            event_emb:   (N, hidden_dim) event embeddings

        Returns:
            (N,) probabilities
        """
        combined = torch.cat([athlete_emb, event_emb], dim=-1)
        logits = self.link_predictor(combined).squeeze(-1)
        return torch.sigmoid(logits)

    def compute_loss(
        self,
        data: HeteroData,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BCE loss for link prediction.

        Args:
            data: HeteroData with node features and edge indices
            pos_edge_index: (2, num_pos) positive edge indices (actual medals)
            neg_edge_index: (2, num_neg) negative edge indices (no medal)

        Returns:
            Scalar loss
        """
        x_dict = self.forward(data)

        # Positive edges
        pos_athlete_emb = x_dict["athlete"][pos_edge_index[0]]
        pos_event_emb = x_dict["event"][pos_edge_index[1]]
        pos_pred = self.predict_link(pos_athlete_emb, pos_event_emb)

        # Negative edges
        neg_athlete_emb = x_dict["athlete"][neg_edge_index[0]]
        neg_event_emb = x_dict["event"][neg_edge_index[1]]
        neg_pred = self.predict_link(neg_athlete_emb, neg_event_emb)

        # Labels
        pos_labels = torch.ones_like(pos_pred)
        neg_labels = torch.zeros_like(neg_pred)

        # Combine
        all_pred = torch.cat([pos_pred, neg_pred])
        all_labels = torch.cat([pos_labels, neg_labels])

        return F.binary_cross_entropy(all_pred, all_labels)


# ── Factory Function ─────────────────────────────────

def build_model(metadata: tuple) -> OlympusHeteroGNN:
    """
    Build and return an OlympusHeteroGNN given graph metadata.

    Args:
        metadata: (node_types, edge_types) from HeteroData.metadata()

    Returns:
        OlympusHeteroGNN instance
    """
    model = OlympusHeteroGNN(
        hidden_dim=GNN_HIDDEN_DIM,
        num_layers=GNN_NUM_LAYERS,
        dropout=GNN_DROPOUT,
        metadata=metadata,
    )
    return model
