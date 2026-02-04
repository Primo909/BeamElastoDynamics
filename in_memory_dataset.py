import torch
from torch_geometric.data import Dataset
import os


class InMemoryTimeStepDataset(Dataset):
    """In-memory dataset."""

    def __init__(self, sample_dir):
        self.data = []
        sample_files = sorted(
            [
                os.path.join(sample_dir, f)
                for f in os.listdir(sample_dir)
                if f.startswith("graph_") and f.endswith(".pt")
            ]
        )
        print(f"Found {len(sample_files)} sample files")
        for sample_file in sample_files:
            sample_data = torch.load(
                sample_file, map_location="cpu", weights_only=False
            )
            self.data.extend(sample_data)  # Flatten all time steps into one list
        print(f"Loaded the dataset with {len(self.data)} samples")

    def __getitem__(self, idx):
        return self.data[
            idx
        ]  # dict with graph, mesh_edge_features, world_edge_features

    def __len__(self):
        return len(self.data)


class LazyTimeStepDataset(torch.utils.data.Dataset):
    """Lazy dataset that loads graphs from disk on demand, avoiding loading
    the entire dataset into memory at once."""

    def __init__(self, sample_dir, num_time_steps):
        self.sample_files = sorted(
            [
                os.path.join(sample_dir, f)
                for f in os.listdir(sample_dir)
                if f.startswith("graph_") and f.endswith(".pt")
            ]
        )
        self.num_steps = num_time_steps - 1
        self.total_samples = len(self.sample_files) * self.num_steps
        print(
            f"Found {len(self.sample_files)} sample files, "
            f"{self.num_steps} steps each, "
            f"{self.total_samples} samples in total."
        )

    def __getitem__(self, idx):
        file_idx = idx // self.num_steps
        idx_in_file = idx % self.num_steps
        sample_file = self.sample_files[file_idx]
        sample_data = torch.load(sample_file, map_location="cpu", weights_only=False)
        return sample_data[idx_in_file]

    def __len__(self):
        return self.total_samples
