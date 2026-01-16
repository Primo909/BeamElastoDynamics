# %%
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# %%
from in_memory_dataset import InMemoryTimeStepDataset

# %% [markdown]
# ### Load Dataset


train_dataset = InMemoryTimeStepDataset(sample_dir="dataset/beam/train")
test_dataset = InMemoryTimeStepDataset(sample_dir="dataset/beam/test")

batch_size = 4
num_workers = 0
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %% [markdown]
# ### Initialize Model

# %%
from models.vinay_mgn import MeshGraphNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

node_channels = 10
edge_channels = 4
num_messages = 8
latent_dim = 128

model = MeshGraphNet(
    node_channels=node_channels,
    edge_channels=edge_channels,
    latent_size=latent_dim,
    num_msgs=num_messages,
)
_ = model.to(device)

# %% [markdown]
# ### Initialize Trainer

# %%
from trainer import Trainer

lr = 3e-5
loss_type = "mse"

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
trainer = Trainer(model, optimizer, device, loss_type=loss_type)
torch.cuda.empty_cache()

# %% [markdown]
# ### Train Loop

import json

from preprocess_data import Stats

# %%


print("Loading stats...")
with open("Results/train/stats/stats.json", "r") as f:
    stats = json.load(f)
node_stats = Stats.from_dict(stats["node"])
edge_stats = Stats.from_dict(stats["edge"])
target_stats = Stats.from_dict(stats["target"])
dt = 0.08

total_epochs = 50

epoch_bar = tqdm(range(total_epochs), desc="Epochs")
for epoch in epoch_bar:
    for batch in tqdm(train_dataloader, desc="Batches", leave=False):
        # print("hey")
        trainer.train(batch)

    trainer.test(
        test_loader=test_dataloader,
        node_stats=node_stats,
        edge_stats=edge_stats,
        target_stats=target_stats,
        dt=dt,
    )
    trainer.epoch_end()
    trainer.save_model()
    epoch_bar.set_postfix(
        {"last_loss": f"{trainer.loss:.6f}, gen_loss: {trainer.gen_loss:.6f}"}
    )


# %% [markdown]
# ### Show Training Progress

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))
plt.plot(trainer.train_history)
plt.show()

# %% [markdown]
# ### Show animation

# %%
from rollout_utils import do_rollout

error_x, error_v, error_stress, true_rollout, pred_rollout = do_rollout(
    model=trainer.model,
    test_loader=test_dataloader,
    device=trainer.device,
    node_stats=node_stats,
    edge_stats=edge_stats,
    target_stats=target_stats,
    dt=dt,
)

# %%
from make_gif import make_beam_comparison_gif

make_beam_comparison_gif(
    pred_rollout=pred_rollout,
    true_rollout=true_rollout,
    L=1.0,
    W=0.1,
    D=0.04,  # Beam dimensions
    out_gif="beam_comparison.gif",
    fps=4,
)

# %%
from IPython.display import Image

Image(filename="beam_comparison.gif")

# %%
