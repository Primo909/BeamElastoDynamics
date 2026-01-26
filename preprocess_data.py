### feature engineeringo
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from gpytoolbox import volume
from torch import Tensor
from torch_geometric.data import Batch, Data

from utils import compute_von_mises_3d


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


def calculate_kinematics(x: Tensor, dt: float) -> tuple[Tensor, Tensor]:
    v0 = torch.zeros(*x[0].shape, device=x.device, dtype=x.dtype).unsqueeze(0)
    dx = x[1:] - x[:-1]
    v_steps = dx / dt
    v = torch.cat([v0, v_steps], dim=0)
    dv = v[1:] - v[:-1]
    a = dv / dt
    return v, a


def build_valued_node_features(
    x: Tensor, v: Tensor, u: Tensor, node_force: Tensor
) -> Tensor:
    """Build valued node features: [position, velocity, rest_position, node_force]"""
    return torch.concat([x, v, u, node_force], dim=1)


def build_edge_features(x: Tensor, u: Tensor, edge_index: Tensor) -> Tensor:
    """Build edge features: [relative_position, relative_position_norm, rest_relative_position, rest_relative_position_norm]"""
    src, dst = edge_index
    xij = x[dst] - x[src]
    uij = u[dst] - u[src]
    xij_norm = torch.norm(xij, dim=1).unsqueeze(1)
    uij_norm = torch.norm(uij, dim=1).unsqueeze(1)
    return torch.concat([xij, xij_norm, uij, uij_norm], dim=1)


def load_stats(
    stats_path: Path = Path("./Results/train/stats/stats.json"),
) -> tuple[Stats, Stats, Stats]:
    """Load normalization statistics from JSON file"""
    with open(stats_path, "r") as f:
        stats_dict = json.load(f)
    node_stats = Stats.from_dict(stats_dict["node"])
    edge_stats = Stats.from_dict(stats_dict["edge"])
    target_stats = Stats.from_dict(stats_dict["target"])
    return node_stats, edge_stats, target_stats


def preprocess_data(
    data_dir: Path,
    split: str,
    noise_scale: float = 0.0,
    dt=0.08,
    recalc_velocities: bool = False,
    target_dir: Path = Path("dataset/beam/"),
):
    SEED = 42
    torch.manual_seed(SEED)

    if True:
        data_dir = data_dir / f"{split}" / "graphs"
        filenames = list(data_dir.glob("*.pt"))

        out_graphs = []
        categorical_node_features = []
        valued_node_features = []
        edge_features = []
        traj_idx = []
        targets = []
        if split != "train":
            node_stats, edge_stats, target_stats = load_stats()
        for idx, filename in enumerate(filenames):
            print(f"Processing file: {filename} ({idx + 1}/{len(filenames)})")
            dataloader = torch.load(filename, weights_only=False)
            graphs = []
            for data in dataloader:
                graph = data.to("cpu")
                graphs.append(graph)
            whole_trajectory_batch = Batch.from_data_list(graphs)
            len_traj = len(graphs)

            x = whole_trajectory_batch.x.reshape(len_traj, -1, 3)
            v, a = calculate_kinematics(x, dt)

            if split == "train":
                noise = noise_scale * torch.randn_like(v)
                x = x + noise
                v = v - noise / dt  # adjust velocity accordingly

            v_next = v[1:]
            x_next = x[1:]

            u = x[0]  # mesh position is just mesh at rest
            all_node_force = whole_trajectory_batch.x_node_force.reshape(
                len_traj, -1, 3
            )
            boundary_condition = graphs[
                0
            ].x_bc  # 1 fixed, 0 free, does not change over time
            for t, graph in enumerate(graphs):
                if t > 48:
                    break
                traj_idx.append(idx)
                _node_force = all_node_force[t]
                noise = noise_scale * torch.randn_like(x)
                _x = x[t]
                _v = v[t]
                _a = a[t]
                cells = graph.x_element_connectivity[0]
                Volume = np.abs(volume(_x.numpy(), cells))
                next_Volume = np.abs(volume(x_next[t].numpy(), cells))
                _categorical_node_attr = boundary_condition
                _node_attr = build_valued_node_features(_x, _v, u, _node_force)
                _edge_attr = build_edge_features(_x, u, graph.edge_index)

                von_mises = torch.log(
                    compute_von_mises_3d(graph.y_sig_t).unsqueeze(1) + 1e-6
                )

                if split != "train":
                    edge_attr = normalize(_edge_attr, edge_stats)
                    vector_node_attr = normalize(_node_attr, node_stats)
                    categorical_node_attr = _categorical_node_attr
                    node_attr = torch.concat(
                        [categorical_node_attr, vector_node_attr], dim=1
                    )

                    out_graphs.append(
                        Data(
                            x=node_attr,
                            edge_attr=edge_attr,
                            edge_index=graph.edge_index,
                            # additional eval info
                            t=graph.time,
                            x_pos_t=x[t],
                            x_vel_t=v[t],
                            node_force=all_node_force[t],
                            node_force_next=all_node_force[t + 1],
                            x_next=x_next[t],
                            v_next=v_next[t],
                            y_acc_t=a[t],
                            y_von_mises=von_mises,
                            x_element_connectivity=graph.x_element_connectivity,
                            u=u,
                        )
                    )
                else:
                    categorical_node_features.append(boundary_condition)
                    valued_node_features.append(
                        build_valued_node_features(_x, _v, u, _node_force)
                    )
                    edge_features.append(build_edge_features(_x, u, graph.edge_index))
                    out_graphs.append(
                        Data(
                            edge_index=graph.edge_index,
                            t=graph.time,
                            y_acc_t=_a,
                            x_pos_t=_x,
                            x_vel_t=_v,
                            volume=Volume,
                            next_volume=next_Volume,
                            cells=graph.x_element_connectivity[0],
                        )
                    )
                    targets.append(torch.concat([_a, von_mises], dim=1))
        if split == "train":
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

            for t in range(len(out_graphs)):
                _valued_node_features = normalize(valued_node_features[t], node_stats)
                _edge_features = normalize(edge_features[t], edge_stats)
                _targets = normalize(targets[t], target_stats)
                out_graphs[t].x = torch.concat(
                    [categorical_node_features[t], _valued_node_features], dim=1
                )
                out_graphs[t].edge_attr = _edge_features
                out_graphs[t].y = _targets
            ### save file

        save_dir = Path(f"{target_dir}/{split}")
        save_dir.mkdir(parents=True, exist_ok=True)
        for traj in set(traj_idx):
            print(f"Processing trajectory: {traj}")
            graphs = [graph for idx, graph in zip(traj_idx, out_graphs) if idx == traj]
            print(f"Length of trajectory {traj}: {len(graphs)}")
            file_path = save_dir / f"graph_{traj:02d}.pt"
            torch.save(graphs, file_path)


if __name__ == "__main__":
    from pathlib import Path

    print("Preprocessing training data...")
    preprocess_data(
        data_dir=Path("Results/"),
        split="train",
        noise_scale=0.0001,
        recalc_velocities=True,
    )

    print()
    print("Preprocessing test data...")
    preprocess_data(data_dir=Path("Results/"), split="val", recalc_velocities=True)
