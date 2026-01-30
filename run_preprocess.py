import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from preprocess_data import preprocess_data

parser = argparse.ArgumentParser(description="Download and preprocess beam elastodynamics data")
parser.add_argument(
    "--dataset",
    type=str,
    choices=["small", "medium", "large"],
    default="small",
    help="Dataset size to preprocess: small, medium, or large (default: small)",
)
parser.add_argument(
    "--noise-scale",
    type=float,
    default=0.0003,
    help="Noise scale for training data augmentation (default: 0.0003)",
)
parser.add_argument(
    "--skip-download",
    action="store_true",
    help="Skip downloading from HuggingFace (use if data is already downloaded)",
)
args = parser.parse_args()

base_dir = Path("./elastic-beam-3d")
dataset_size = args.dataset

# Download data from HuggingFace
if not args.skip_download:
    repo_id = "kevinsteiner/elastic-beam-3d"
    print(f"Downloading dataset from {repo_id}...")
    ignore = []
    if dataset_size != "large":
        ignore.append("raw/large/**")
    if dataset_size != "medium":
        ignore.append("raw/medium/**")
    if dataset_size != "small":
        ignore.append("raw/small/**")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(base_dir),
        ignore_patterns=ignore,
    )
    print(f"Dataset downloaded to {base_dir}")

raw_data_dir = base_dir / "raw" / "single-direction-beam" / dataset_size
target_dir = base_dir / "processed" / dataset_size
stats_dir = target_dir / "train" / "stats"

print(f"Preprocessing {dataset_size} dataset")
print(f"  Raw data: {raw_data_dir}")
print(f"  Output:   {target_dir}")
print(f"  Stats:    {stats_dir}")
print()

# Train split (must be first so stats are computed)
print("1. Preprocessing training data...")
preprocess_data(
    data_dir=raw_data_dir,
    split="train",
    noise_scale=args.noise_scale,
    recalc_velocities=True,
    target_dir=target_dir,
    stats_dir=stats_dir,
)

# Validation split
print()
print("2. Preprocessing validation data...")
preprocess_data(
    data_dir=raw_data_dir,
    split="val",
    recalc_velocities=True,
    target_dir=target_dir,
    stats_dir=stats_dir,
)

# Test split (if available)
test_graphs_dir = raw_data_dir / "test" / "graphs"
if test_graphs_dir.exists() and any(test_graphs_dir.glob("*.pt")):
    print()
    print("3. Preprocessing test data...")
    preprocess_data(
        data_dir=raw_data_dir,
        split="test",
        recalc_velocities=True,
        target_dir=target_dir,
        stats_dir=stats_dir,
    )
else:
    print()
    print("3. No test data found, skipping.")

print()
print("Preprocessing complete!")
