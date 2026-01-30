import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from preprocess_data import preprocess_data


def parse_volume_loss_weight(value):
    """Parse volume loss weight argument - accepts 'false' or a float value."""
    if value.lower() == "false":
        return False
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"volume_loss_weight must be 'false' or a float value, got: {value}"
        )


parser = argparse.ArgumentParser(
    description="Train MeshGraphNet with optional volume loss weight"
)
parser.add_argument(
    "--volume-loss-weight",
    type=parse_volume_loss_weight,
    default=False,
    help="Weight for volume loss (use 'false' to disable or provide a float value)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=300,
    help="Total number of training epochs (default: 300)",
)
parser.add_argument(
    "--resume-from",
    type=str,
    default=None,
    help="Path to checkpoint file to resume training from (e.g., saved_models/2026-01-29_07-51-59/Epoch_480_GenLoss_0.0116324676.pth)",
)
parser.add_argument(
    "--do-full-volume",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use full volume computation (default: True). Use --no-do-full-volume to disable.",
)
args = parser.parse_args()

DATASET = "large"

if DATASET == "large":
    src_folder = Path("Results_large/")
    target_folder = Path("dataset/beam_large/")
else:
    raise ValueError(f"Unknown DATASET: {DATASET}")
print("Preprocessing training data...")
# preprocess_data(
#     data_dir=src_folder,
#     split="train",
#     noise_scale=0.0003,
#     recalc_velocities=True,
#     target_dir=target_folder,
# )

# print()
# print("Preprocessing test data...")
# preprocess_data(
#     data_dir=src_folder, split="val", recalc_velocities=True, target_dir=target_folder
# )

from torch_geometric.loader import DataLoader

from in_memory_dataset import InMemoryTimeStepDataset

if DATASET == "large":
    target_folder = Path("dataset/beam_large/")
else:
    raise ValueError(f"Unknown DATASET: {DATASET}")
train_dataset = InMemoryTimeStepDataset(sample_dir=target_folder / "train")
test_dataset = InMemoryTimeStepDataset(sample_dir=target_folder / "val")

batch_size = 16
num_workers = 4
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

from models.vinay_mgn import MeshGraphNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

node_channels = 13
edge_channels = 8
num_messages = 8
latent_dim = 128

torch.manual_seed(42)
model = MeshGraphNet(
    node_channels=node_channels,
    edge_channels=edge_channels,
    latent_size=latent_dim,
    num_msgs=num_messages,
)
_ = model.to(device)

from trainer import Trainer

lr = 5e-5
loss_type = "mse"

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
trainer = Trainer(model, optimizer, device, loss_type=loss_type, use_wandb=True)
torch.cuda.empty_cache()
print(f"Training_id: {trainer.training_id}")
print(f"Volume loss weight: {args.volume_loss_weight}")


import json

from preprocess_data import Stats

print("Loading stats...")
with open("Results/train/stats/stats.json", "r") as f:
    stats = json.load(f)
node_stats = Stats.from_dict(stats["node"])
edge_stats = Stats.from_dict(stats["edge"])
target_stats = Stats.from_dict(stats["target"])
dt = 0.08

total_epochs = args.epochs
validation_interval = 20

# Handle checkpoint resumption
start_epoch = 0
if args.resume_from:
    resume_epoch = trainer.load_checkpoint(args.resume_from)
    start_epoch = resume_epoch + 1
    print(f"Resuming training from epoch {start_epoch}")

print(f"Total epochs: {total_epochs}")
print(f"Starting from epoch: {start_epoch}")

epoch_bar = tqdm(range(start_epoch, total_epochs), desc="Epochs", initial=start_epoch, total=total_epochs)
for epoch in epoch_bar:
    for batch in tqdm(train_dataloader, desc="Batches", leave=False):
        # print("hey")
        batch.to(device)
        if epoch < 300:
            volume_loss = False
        else:
            volume_loss = args.volume_loss_weight
        trainer.train(
            batch,
            node_stats=node_stats,
            edge_stats=edge_stats,
            target_stats=target_stats,
            volume_loss_weight=volume_loss,
            do_full_volume=args.do_full_volume,
        )

    if epoch % validation_interval == 0:
        trainer.test(
            test_loader=test_dataloader,
            node_stats=node_stats,
            edge_stats=edge_stats,
            target_stats=target_stats,
            dt=dt,
            epoch=epoch,
        )
        trainer.save_model(epoch=epoch)
    trainer.epoch_end(epoch=epoch)
    epoch_bar.set_postfix(
        {
            "last_loss": f"{trainer.loss:.6f}, full_rollout_error: {trainer.rollout_all_step_error:.6f}",
        }
    )
