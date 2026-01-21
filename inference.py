import argparse
from pathlib import Path

import torch

from make_gif import make_beam_comparison_gif
from models.vinay_mgn import MeshGraphNet
from preprocess_data import Stats
from rollout_utils import do_rollout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Run inference with a trained model")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="best_model.pth",
    help="Path to model checkpoint file",
)
args = parser.parse_args()


checkpoint_path = Path(args.checkpoint)

node_channels = 13
edge_channels = 8
num_messages = 8
latent_dim = 128

dt = 0.08

torch.manual_seed(42)
model = MeshGraphNet(
    node_channels=node_channels,
    edge_channels=edge_channels,
    latent_size=latent_dim,
    num_msgs=num_messages,
)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

no_rollout = False


###
import json

from torch_geometric.loader import DataLoader

from in_memory_dataset import InMemoryTimeStepDataset

test_dataset = InMemoryTimeStepDataset(sample_dir="dataset/beam/val")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
with open("Results/train/stats/stats.json", "r") as f:
    stats = json.load(f)
node_stats = Stats.from_dict(stats["node"])
edge_stats = Stats.from_dict(stats["edge"])
target_stats = Stats.from_dict(stats["target"])

true_rollout, pred_rollout = do_rollout(
    model=model,
    test_loader=test_dataloader,
    device=device,
    node_stats=node_stats,
    edge_stats=edge_stats,
    target_stats=target_stats,
    dt=dt,
    skip_first=0,
    rollout_steps=30,
    dont_rollout=no_rollout,
)


make_beam_comparison_gif(
    pred_rollout=pred_rollout,
    true_rollout=true_rollout,
    L=1.0,
    W=0.1,
    D=0.04,  # Beam dimensions
    out_gif=f"beam_comparison_{checkpoint_path.name}.gif",
    fps=4,
)
