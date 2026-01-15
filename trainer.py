import os
from pathlib import Path
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
from tqdm import tqdm
import random
import pickle
import shutil
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from in_memory_dataset import InMemoryTimeStepDataset
from models.vinay_mgn import MeshGraphNet
from typing_extensions import Literal


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
            print(f"Falling back to MSE loss.")
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

    def test(self, test_loader, gen_roll_out=False, experiment_name="Val"):
        self.model.eval()  # Set model to evaluation mode
        # Store overall test loss history
        # self.gen_test_history.append(
        #     self.gen_loss_pos + self.gen_loss_vel + self.gen_loss_stress
        # )

    def save_model(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # Save model using the generalization loss as part of the filename
        filename = f"GenLoss_{self.gen_loss_pos:.2f}m_{self.gen_loss_vel:.2f}mps_{self.gen_loss_stress:.2f}Nms2.pth"
        self.path = self.model_dir / filename
        torch.save(self.model.state_dict(), self.path)
