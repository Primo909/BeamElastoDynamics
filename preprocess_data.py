import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.nn import Sequential, Linear, ReLU, LayerNorm
import torch.nn.functional as F
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import imageio
from tqdm.notebook import tqdm
import random
import pickle
import shutil

### feature engineeringo
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
from torch_geometric.data import Data, Batch
from beam_fea_solver import compute_von_mises_3d


@dataclass
class Stats:
    mean: torch.Tensor
    std: torch.Tensor

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(mean=torch.tensor(d["mean"]), std=torch.tensor(d["std"]))


def compute_stats(tensor_list) -> Stats:
    # would also be an option to batch over all tensors
    # Batch.from_data_list(train_graphs).x.mean(dim=0)
    # but this uses less memory

    sum = 0
    sum_sq = 0
    count = 0
    for tensor in tensor_list:
        sum += tensor.sum(dim=0)
        sum_sq += (tensor**2).sum(dim=0)
        count += tensor.shape[0]
    mean = sum / count
    var = (sum_sq / count) - (mean**2)
    std = torch.sqrt(var)
    return Stats(mean=mean, std=std)


def normalize(tensor: torch.Tensor, stats: Stats, eps: float = 1e-8) -> torch.Tensor:
    return (tensor - stats.mean) / (stats.std + eps)


def denormalize(tensor: torch.Tensor, stats: Stats, eps: float = 1e-8) -> torch.Tensor:
    return tensor * (stats.std + eps) + stats.mean


if __name__ == "__main__":
    data_dir = Path("Results/train/graphs")
    split = "test"

    if split == "train":
        filenames = list(data_dir.glob("*.pt"))

        train_graphs = []
        categorical_node_features = []
        valued_node_features = []
        edge_features = []
        traj_idx = []
        targets = []
        for idx, filename in enumerate(filenames):
            dataloader = torch.load(filename, weights_only=False)
            graphs = []
            for data in dataloader:
                graph = data.to("cpu")
                graphs.append(graph)
            for graph in graphs:
                traj_idx.append(idx)
                boundary_condition = graph.x_bc  # 1 fixed, 0 free
                node_force = graph.x_node_force  # external force on node
                x = graph.x

                categorical_node_features.append(boundary_condition)
                valued_node_features.append(
                    torch.concat([x, graph.x_vel_t, node_force], dim=1)
                )
                print(graph.time)
                train_graphs.append(
                    Data(
                        edge_index=graph.edge_index,
                        t=graph.time,
                        y_acc_t=graph.y_acc_t,
                        y_pos_t=graph.x,
                        y_vel_t=graph.x_vel_t,
                    )
                )

                src, dst = graph.edge_index
                xij = x[dst] - x[src]
                xij_norm = torch.norm(xij, dim=1).unsqueeze(1)
                edge_features.append(torch.concat([xij, xij_norm], dim=1))

                von_mises = compute_von_mises_3d(graph.y_sig_t).unsqueeze(1)
                targets.append(torch.concat([graph.y_acc_t, von_mises], dim=1))

        node_stats = compute_stats(valued_node_features)
        edge_stats = compute_stats(edge_features)
        target_stats = compute_stats(targets)

        stats = {
            "node": node_stats.to_dict(),
            "edge": edge_stats.to_dict(),
            "target": target_stats.to_dict(),
        }
        stats_dir = Path("./Results/train/stats")
        stats_dir.mkdir(parents=True, exist_ok=True)
        with open(stats_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)

        for i in range(len(train_graphs)):
            _valued_node_features = normalize(valued_node_features[i], node_stats)
            _edge_features = normalize(edge_features[i], edge_stats)
            _targets = normalize(targets[i], target_stats)
            train_graphs[i].x = torch.concat(
                [categorical_node_features[i], _valued_node_features], dim=1
            )
            train_graphs[i].edge_attr = _edge_features
            train_graphs[i].y = _targets
        ### save file

        save_dir = Path("dataset/beam/train")
        save_dir.mkdir(parents=True, exist_ok=True)
        for traj in set(traj_idx):
            print(f"Processing trajectory: {traj}")
            graphs = [
                graph for idx, graph in zip(traj_idx, train_graphs) if idx == traj
            ]
            print(f"Length of trajectory {traj}: {len(graphs)}")
            file_path = save_dir / f"graph_{traj:02d}.pt"
            torch.save(graphs, file_path)
    elif split == "test":
        # only load one trajectory
        filenames = list(data_dir.glob("*.pt"))
        filename = filenames[0]
        dataloader = torch.load(filename, weights_only=False)
        graphs = []
        for data in dataloader:
            graph = data.to("cpu")
            graphs.append(graph)
            whole_trajectory_batch = Batch.from_data_list(graphs)
        len_traj = len(dataloader)
        x_next = whole_trajectory_batch.x.reshape(len_traj, -1, 3)[1:]
        v_next = whole_trajectory_batch.x_vel_t.reshape(len_traj, -1, 3)[1:]

        with open("./Results/train/stats/stats.json", "r") as f:
            stats_dict = json.load(f)
        node_stats = Stats.from_dict(stats_dict["node"])
        edge_stats = Stats.from_dict(stats_dict["edge"])
        target_stats = Stats.from_dict(stats_dict["target"])

        test_graphs = []
        for graph in graphs:
            boundary_condition = graph.x_bc  # 1 fixed, 0 free
            node_force = graph.x_node_force  # external force on node
            x = graph.x

            categorical_node_features.append(boundary_condition)
            valued_node_features.append(
                torch.concat([x, graph.x_vel_t, node_force], dim=1)
            )
            print(graph.time)
            test_graphs.append(
                Data(
                    edge_index=graph.edge_index,
                    t=graph.time,
                    y_acc_t=graph.y_acc_t,
                    y_pos_t=graph.x,
                    y_vel_t=graph.x_vel_t,
                    node_force=node_force,
                )
            )

            src, dst = graph.edge_index
            xij = x[dst] - x[src]
            xij_norm = torch.norm(xij, dim=1).unsqueeze(1)
            edge_features.append(torch.concat([xij, xij_norm], dim=1))

            von_mises = compute_von_mises_3d(graph.y_sig_t).unsqueeze(1)
            targets.append(torch.concat([graph.y_acc_t, von_mises], dim=1))
