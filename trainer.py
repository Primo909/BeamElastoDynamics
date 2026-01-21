import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing_extensions import Literal

from preprocess_data import Stats
from rollout_utils import do_rollout, mae


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
        self.training_id = time.strftime("%Y-%m-%d_%H-%M-%S")
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
        self.train_acc_history = []
        self.train_stress_history = []
        self.extr_test_history = []
        self.gen_test_history = []

        # Create a directory for saving trained models
        self.cur_dir = os.getcwd()
        self.model_dir = Path(self.cur_dir) / "saved_models" / self.training_id
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.gen_loss = 0.0  # Initialize generalization loss
        self.epoch_losses = []
        self.train_acc_loss = []
        self.train_stress_loss = []

    def train(self, graph: Data):
        self.model.train()
        # Forward pass through the model
        pred: torch.Tensor = self.model(graph)
        target: torch.Tensor = graph.y.to(self.device)
        # Compute MSE loss
        loss = self.loss_fn(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        pred_acc = pred[:, :3].contiguous()
        target_acc = target[:, :3].contiguous()
        pred_stress = pred[:, 3].contiguous()
        target_stress = target[:, 3].contiguous()

        stress_loss = self.loss_fn(pred_stress, target_stress)
        acc_loss = self.loss_fn(pred_acc, target_acc)

        # loss = acc_loss + stress_loss

        # Store loss history
        self.train_acc_loss.append(acc_loss.cpu().detach().numpy())
        self.train_stress_loss.append(stress_loss.cpu().detach().numpy())

        self.loss = loss.cpu().detach().numpy()
        self.train_history.append(self.loss)
        self.gen_loss_pos = 0.0
        self.gen_loss_vel = 0.0
        self.gen_loss_stress = 0.0
        self.epoch_losses.append(self.loss)

        self.history_full_rollout_loss = []

    def epoch_end(self):
        self.train_acc_history.append(self.train_acc_loss)
        self.train_stress_history.append(self.train_stress_loss)

        self.train_acc_loss = []
        self.train_stress_loss = []
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
        true_rollout, pred_rollout = do_rollout(
            model=self.model,
            test_loader=test_loader,
            device=self.device,
            node_stats=node_stats,
            edge_stats=edge_stats,
            target_stats=target_stats,
            dt=dt,
        )
        rollout_error = mae(true_rollout[-1].x_pos_t, pred_rollout[-1].x_pos_t)

        # Compute generalization errors
        self.rollout_all_step_error = rollout_error

        self.history_full_rollout_loss.append(self.rollout_all_step_error)

    def save_model(self, epoch: int = None):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # Save model using the generalization loss as part of the filename
        if epoch is not None:
            filename = f"Epoch_{epoch}_GenLoss_{self.rollout_all_step_error:.10f}.pth"
        else:
            filename = f"GenLoss_{self.rollout_all_step_error:.10f}.pth"
        self.path = self.model_dir / filename
        torch.save(self.model.state_dict(), self.path)
