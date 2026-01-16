import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing_extensions import Literal

from preprocess_data import Stats
from rollout_utils import do_rollout


class Trainer:
    """
    Trainer class for training and evaluating the MeshGraphNet model.

    This class manages:
    - Training process using graph-based data.
    - Model evaluation through test rollouts.
    - Model checkpointing and saving.

    Parameters:
    - model (torch.nn.Module): The GNN model to be trained.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - device (torch.device): Target computing device (CPU/GPU).
    - train_stats (tuple): Normalization statistics (mean, std) for input/output features.
    """

    def __init__(
        self, model, optimizer, device, loss_type: Literal["mse", "mae"] = "mse"
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        if loss_type == "mse":
            self.loss_fn = F.mse_loss
        elif loss_type == "mae":
            self.loss_fn = F.l1_loss
        else:
            print(f"Unsupported loss function: {loss_type}!")
            print("Falling back to MSE loss.")
            self.loss_fn = F.mse_loss
        # Training and test history
        self.train_history = []
        self.extr_test_history = []
        self.gen_test_history = []

        # Create a directory for saving trained models
        self.cur_dir = os.getcwd()
        self.model_dir = Path(self.cur_dir) / "saved_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.gen_loss = 0.0  # Initialize generalization loss
        self.epoch_losses = []

    def train(self, graph: Data):
        self.model.train()
        # Forward pass through the model
        pred = self.model(graph)
        target = graph.y.to(self.device)

        # Compute MSE loss
        loss = self.loss_fn(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store loss history
        self.loss = loss.cpu().detach().numpy()
        self.train_history.append(self.loss)
        self.gen_loss_pos = 0.0
        self.gen_loss_vel = 0.0
        self.gen_loss_stress = 0.0
        self.epoch_losses.append(self.loss)

    def epoch_end(self):
        self.loss = np.mean(self.epoch_losses)
        self.epoch_losses = []

    def test(
        self,
        test_loader,
        node_stats: Stats,
        edge_stats: Stats,
        target_stats: Stats,
        dt: float,
    ):
        self.model.eval()  # Set model to evaluation modes

        # Perform model rollout testing
        error_x, error_v, error_stress, true_rollout, pred_rollout = do_rollout(
            model=self.model,
            test_loader=test_loader,
            device=self.device,
            node_stats=node_stats,
            edge_stats=edge_stats,
            target_stats=target_stats,
            dt=dt,
        )

        # Compute generalization errors
        self.gen_loss_pos = np.sum(error_x) / len(error_x)
        self.gen_loss_vel = np.sum(error_v) / len(error_v)
        self.gen_loss_stress = np.sum(error_stress) / len(error_stress)

        # Store overall test loss history
        self.gen_test_history.append(
            self.gen_loss_pos + self.gen_loss_vel + self.gen_loss_stress
        )

    def save_model(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # Save model using the generalization loss as part of the filename
        filename = f"GenLoss_{self.gen_loss_pos:.2f}m_{self.gen_loss_vel:.2f}mps_{self.gen_loss_stress:.2f}Nms2.pth"
        self.path = self.model_dir / filename
        torch.save(self.model.state_dict(), self.path)
