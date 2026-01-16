import os
import shutil
import time
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch_geometric.data import Data


def compute_von_mises_3d(stress_9):
    sxx = stress_9[:, 0]
    sxy = stress_9[:, 1]
    sxz = stress_9[:, 2]
    syx = stress_9[:, 3]
    syy = stress_9[:, 4]
    syz = stress_9[:, 5]
    szx = stress_9[:, 6]
    szy = stress_9[:, 7]
    szz = stress_9[:, 8]
    vm = 0.5 * (
        (sxx - syy) ** 2
        + (syy - szz) ** 2
        + (szz - sxx) ** 2
        + 6.0 * (sxy**2 + syz**2 + sxz**2)
    )
    return np.sqrt(vm)


from dataclasses import dataclass


@dataclass
class RolloutGraph(Data):
    x_pos_t: torch.Tensor
    x_vel_t: torch.Tensor
    von_mises: torch.Tensor
    x_element_connectivity: torch.Tensor


def make_beam_comparison_gif(
    pred_rollout: list[RolloutGraph],
    true_rollout: list[RolloutGraph],
    L: float = 1.0,
    W: float = 0.1,
    D: float = 0.04,
    out_gif: str = "beam_comparison.gif",
    fps: int = 4,
    bbox_only_on_true: bool = True,
):
    assert len(pred_rollout) == len(true_rollout), (
        "Mismatch in predicted and ground truth frames."
    )

    time_start = time.time()
    temp_dir = "temp_frames_" + str(time_start)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    frames = []

    # Extract element connectivity from the first graph: it's an Nx4 array showing 4 nodes of the tetrahedral mesh element for each node.
    first_graph = true_rollout[0]
    cells = first_graph.x_element_connectivity[0][
        0
    ]  # it was a mistake that x_element_connectivity was stored as a nested list

    # Find global bounding box across all frames
    all_x, all_y, all_z = [], [], []
    vm_min, vm_max = 0.0, 0.0  # Stress range

    for g_pred, g_true in zip(pred_rollout, true_rollout):
        coords_pred = g_pred.x_pos_t.cpu().numpy()
        coords_true = g_true.x_pos_t.cpu().numpy()
        von_mises_pred = g_pred.von_mises.cpu().numpy()
        von_mises_true = g_true.von_mises.cpu().numpy()

        vm_max = max(vm_max, np.max(von_mises_pred))
        vm_max = max(vm_max, np.max(von_mises_true))
        all_x.extend(coords_true[:, 0])
        all_y.extend(coords_true[:, 1])
        all_z.extend(coords_true[:, 2])

        if bbox_only_on_true:
            continue
        all_x.extend(coords_pred[:, 0])
        all_y.extend(coords_pred[:, 1])
        all_z.extend(coords_pred[:, 2])

    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    z_min, z_max = np.min(all_z), np.max(all_z)

    # Iterate over time steps to create frames
    for i, (graph_pred, graph_true) in enumerate(zip(pred_rollout, true_rollout)):
        print(f"Rendering frame {i}/{len(pred_rollout) - 1}")

        coords_pred = graph_pred.x_pos_t.cpu().numpy()
        coords_true = graph_true.x_pos_t.cpu().numpy()

        vm_pred = graph_pred.von_mises.cpu().numpy()
        vm_true = graph_true.von_mises.cpu().numpy()

        fig, axarr = plt.subplots(
            1, 2, figsize=(16, 8), subplot_kw={"projection": "3d"}
        )

        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=vm_min, vmax=vm_max)

        # Function to plot a beam in 3D with element connectivity
        def plot_beam(ax, coords, vm, title):
            polys = []
            face_values = []
            for tet in cells:
                tet_coords = coords[tet]  # No offset needed for separate subplots
                polys.append(tet_coords[:3])  # Use first 3 vertices for a face
                face_values.append(np.mean(vm[tet]))  # Interpolated stress

            face_values = np.array(face_values)
            face_colors = cmap(norm(face_values))

            polycoll = Poly3DCollection(polys)
            polycoll.set_facecolors(face_colors)
            polycoll.set_edgecolor("none")
            ax.add_collection3d(polycoll)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            ax.set_box_aspect((L * 0.1, 0.8 * W, 2.5 * D))
            ax.view_init(elev=30, azim=45)
            ax.set_title(title)

        # Left subplot: Ground Truth
        plot_beam(axarr[0], coords_true, vm_true, title="Ground Truth")

        # Right subplot: Predicted
        plot_beam(axarr[1], coords_pred, vm_pred, title="Predicted MeshGraphNet")

        cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # (left, bottom, width, height)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
        cbar.set_label("von Mises Stress")
        # cb = fig.colorbar(mappable, ax=axarr.ravel().tolist(), shrink=0.6)
        # cb.set_label("von Mises Stress")

        frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        frame_path = Path(frame_path)
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(frame_path, dpi=100)
        plt.close(fig)

        frames.append(imageio.imread(frame_path))

    # Create GIF
    imageio.mimsave(out_gif, frames, fps=fps, loop=0)
    print(f"GIF saved => {out_gif}")

    # Cleanup
    shutil.rmtree(temp_dir)
    print("Done.")
