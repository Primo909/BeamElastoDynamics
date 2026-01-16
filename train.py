import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from in_memory_dataset import InMemoryTimeStepDataset
from models.vinay_mgn import MeshGraphNet
from trainer import Trainer

# Define model hyperparameters
num_messages = 8  # Number of message-passing steps in the GNN
latent_dim = 128  # Size of the latent feature space
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the MeshGraphNet model
MODEL = MeshGraphNet(
    node_channels=10,
    edge_channels=4,
    latent_size=latent_dim,
    num_msgs=num_messages,
)
MODEL.to(device)
optimizer = torch.optim.Adam(MODEL.parameters(), lr=3e-5)
trainer = Trainer(MODEL, optimizer, device, loss_type="mse")
torch.cuda.empty_cache()

total_epochs = 50
progress_bar = tqdm(total=total_epochs, desc="Training Progress")
data_dir = "dataset/beam/train"
batch_size = 4
shuffle = True
dataset = InMemoryTimeStepDataset(data_dir)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=True,  # Ensures all batches have the same size
    num_workers=0,
)
test_loader = DataLoader(
    dataset[:50],
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=0,
)
epoch_bar = tqdm(range(total_epochs), desc="Epochs")
for epoch in epoch_bar:
    for batch in tqdm(dataloader, desc="Batches", leave=False):
        # print("hey")
        trainer.train(batch)
    trainer.epoch_end()
    trainer.save_model()
    epoch_bar.set_postfix({"last_loss": f"{trainer.loss:.6f}"})

    trainer.test(test_loader=test_loader, gen_roll_out=False, experiment_name="Val")
