import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean


def build_mlp_d(
    in_size,
    hidden_size,
    out_size,
    num_layers=1,
    lay_norm=True,
    use_sigmoid=False,
    use_softmax=False,
):
    """
    Builds a multi-layer perceptron (MLP) with configurable depth and optional layer normalization and sigmoid or softmax activation.

    Args:
        in_size (int): The size of the input feature vector.
        hidden_size (int): The size of the hidden layers.
        out_size (int): The size of the output layer.
        num_layers (int): The number of layers in the MLP.
        lay_norm (bool): Flag to add layer normalization after the last linear layer.
        use_sigmoid (bool): Flag to add a sigmoid activation layer at the output.
        use_softmax (bool): Flag to add a softmax activation layer at the output.

    Returns:
        nn.Module: The constructed MLP model.

    Raises:
        ValueError: If both use_sigmoid and use_softmax are True.
    """
    if use_sigmoid and use_softmax:
        raise ValueError("Only one of use_sigmoid or use_softmax can be true.")

    layers = [nn.Linear(in_size, hidden_size), nn.ReLU()]

    # Add intermediate layers
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())

    # Add the output layer
    layers.append(nn.Linear(hidden_size, out_size))

    # Create the MLP module
    module = nn.Sequential(*layers)

    # Optionally add layer normalization
    if lay_norm:
        module = nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))

    # Optionally add sigmoid activation
    if use_sigmoid:
        module = nn.Sequential(module, nn.Sigmoid())

    # Optionally add softmax activation
    if use_softmax:
        module = nn.Sequential(module, nn.Softmax(dim=-1))

    return module


class Encoder(torch.nn.Module):
    """
    Encoder for node and edge features.

    - Encodes input node and edge features into latent space.
    - Uses Multi-Layer Perceptron (MLP) with layer normalization.

    Parameters:
    - `latent_size`: Size of the latent space.
    """

    def __init__(self, node_channels, edge_channels, latent_size=128):
        super(Encoder, self).__init__()
        # Encode edge features (4 input features → latent space)
        self.edge_encoder = build_mlp_d(
            edge_channels, latent_size, latent_size, num_layers=1, lay_norm=True
        )

        # Encode node features (19 input features → latent space)
        self.node_encoder = build_mlp_d(
            node_channels, latent_size, latent_size, num_layers=1, lay_norm=True
        )

    def forward(self, node_features, edge_features):
        """
        Forward pass: encode node and edge features.

        Returns:
        - `node_embedding`: Encoded node features.
        - `edge_embedding`: Encoded edge features.
        """
        node_embedding = self.node_encoder(node_features)
        edge_embedding = self.edge_encoder(edge_features)
        return node_embedding, edge_embedding


class GraphNetBlock(torch.nn.Module):
    """
    Graph network block for message passing.

    - Updates edge embeddings based on sender and receiver nodes.
    - Aggregates messages at the nodes.
    - Updates node embeddings with aggregated edge information.

    Parameters:
    - `latent_size`: Size of latent feature space.
    - `in_size_edge_MLP`: Input size for edge MLP.
    - `in_size_node_MLP`: Input size for node MLP.
    """

    def __init__(
        self,
        in_size_edge_MLP,
        in_size_node_MLP,
        latent_size=128,
    ):
        super(GraphNetBlock, self).__init__()
        self._latent_size = latent_size

        # Edge MLP: updates edge embeddings based on node interactions
        self.edge_net = build_mlp_d(
            in_size_edge_MLP, latent_size, latent_size, lay_norm=True
        )

        # Node MLP: updates node embeddings based on aggregated edge messages
        self.node_net = build_mlp_d(
            in_size_node_MLP, latent_size, latent_size, lay_norm=True
        )

    def forward(self, edge_index, node_embedding, edge_embedding):
        """
        Forward pass: message passing and node update.

        - `edge_index`: Edge connectivity (senders, receivers).
        - `node_embedding`: Node embeddings.
        - `edge_embedding`: Edge embeddings.

        Returns:
        - Updated node embeddings.
        - Updated edge embeddings.
        """
        senders, receivers = edge_index  # Extract sender and receiver node indices
        num_nodes = node_embedding.shape[0]

        # Compute new edge features using edge MLP
        edge_attr_ = self.edge_net(
            torch.hstack(
                (node_embedding[senders], node_embedding[receivers], edge_embedding)
            )
        )

        # Aggregate received edge messages at each node
        agg_received_edges = scatter_add(
            edge_attr_, receivers.long(), dim=0, dim_size=num_nodes
        )

        # Update node features using aggregated edge messages
        node_attr_ = self.node_net(torch.hstack((agg_received_edges, node_embedding)))

        # Add residual connections for stability
        new_node_embedding = node_attr_ + node_embedding
        new_edge_embedding = edge_attr_ + edge_embedding

        return new_node_embedding, new_edge_embedding


class Decoder(torch.nn.Module):
    """
    Decoder to predict acceleration and stress tensor from node embeddings.

    - Uses two separate MLPs:
      - One for acceleration (`3D output`).
      - One for stress tensor (`6D output`).

    Parameters:
    - `latent_size`: Size of latent feature space.
    """

    def __init__(self, latent_size=128):
        super(Decoder, self).__init__()
        self.acc_decoder = build_mlp_d(latent_size, latent_size, 3, lay_norm=False)
        self.von_mises_decoder = build_mlp_d(
            latent_size, latent_size, 1, lay_norm=False
        )

    def forward(self, node_embedding):
        """
        Forward pass: decode acceleration and stress tensor.

        Returns:
        - `decoded_acc`: Predicted acceleration (3D).
        - `decoded_stress_flat`: Flattened stress tensor (9D).
        """
        decoded_acc = self.acc_decoder(node_embedding)
        decoded_von_mis = self.von_mises_decoder(node_embedding)
        pred = torch.hstack((decoded_acc, decoded_von_mis))
        return pred


class MeshGraphNet(nn.Module):
    """
    MeshGraphNet Model.

    - Encodes node and edge features.
    - Processes them through multiple message-passing layers.
    - Decodes node-level properties (acceleration and stress tensor).

    Parameters:
    - `latent_size`: Size of latent feature space.
    - `num_msgs`: Number of message-passing steps.
    """

    def __init__(
        self,
        node_channels: int,
        edge_channels: int,
        latent_size: int = 128,
        num_msgs=3,
    ):
        super(MeshGraphNet, self).__init__()

        # Encoder for input graph data
        self.encoder = Encoder(
            node_channels=node_channels,
            edge_channels=edge_channels,
            latent_size=latent_size,
        )

        # Message-passing layers (GraphNetBlocks)
        self.processer_list = nn.ModuleList(
            [
                GraphNetBlock(
                    latent_size=latent_size,
                    in_size_edge_MLP=3 * latent_size,
                    # Edge MLP takes sender, receiver, and edge features
                    # phi_e: (h_i, h_j, e_ij) -> e_ij'
                    in_size_node_MLP=2 * latent_size,
                )  # Node update MLP takes aggregated edge messages + current node features
                # phi_v: (sum_j e_ij', h_i) -> h_i'
                for _ in range(num_msgs)
            ]
        )

        # Decoder to predict acceleration and stress tensor
        self.decoder = Decoder(latent_size)

    def forward(self, ingraph):
        """
        Forward pass: Encodes, processes, and decodes graph data.

        Parameters:
        - `ingraph`: Input graph with node features `x` and edge features `edge_attr`.

        Returns:
        - Predicted node accelerations.
        - Predicted stress tensor.
        """
        node_features = ingraph.x
        edge_features = ingraph.edge_attr
        edge_index = ingraph.edge_index

        # Encode node and edge features
        node_embedding, edge_embedding = self.encoder(node_features, edge_features)

        # Process through multiple message-passing layers
        for processor in self.processer_list:
            node_embedding, edge_embedding = processor(
                edge_index, node_embedding, edge_embedding
            )

        # Decode acceleration and stress tensor from node embeddings
        pred = self.decoder(node_embedding)

        return pred  # Output acceleration and stress tensor
