from typing_extensions import Literal
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from preprocess_data import Stats, denormalize, normalize


def mae(target, pred):
    return torch.mean(torch.abs(target - pred))


def do_rollout(
    model: torch.nn.Module,
    test_loader: DataLoader,
    node_stats: Stats,
    edge_stats: Stats,
    target_stats: Stats,
    dt: float,
    device: torch.device,
    skip_first: int = 0,
    skip_after: int = 99999999,
    dont_rollout: bool = False,
    integrator: Literal["semiimplicit_euler", "explicit_euler", "trapezoidal"] = "semiimplicit_euler",
):
    """returns `error_x, error_v, error_stress, true_rollout, pred_rollout`"""
    with torch.no_grad():
        # Store errors for evaluation
        error_x = []
        error_v = []
        error_stress = []
        true_rollout = []
        pred_rollout = []
        for t, graph in enumerate(test_loader):
            u = graph.u
            if t < skip_first:
                continue
            if t > skip_after:
                break
            if t == skip_first:
                input_graph = graph
            if dont_rollout:
                input_graph = graph
            pred = model(input_graph.to(device))
            input_graph = input_graph.cpu()
            pred = pred.cpu()

            pred_denorm = denormalize(pred, target_stats)
            # a_true = graph.y_acc_t
            von_mises_true = graph.y_von_mises

            a = pred_denorm[:, :3]
            von_mises = pred_denorm[:, 3]

            # exclude the first dimension (boundary condition)
            x = input_graph.x_pos_t
            v = input_graph.x_vel_t

            # integrate
            boundary_condition = graph.x[:, 0]  # 1 fixed, 0 free

            v_new = v + a * dt
            x_new = x + v_new * dt 
            x_new[boundary_condition == 1] = x[boundary_condition == 1]
            v_new[boundary_condition == 1] = 0.0

            # Use new graph to calculate error
            x_true = graph.x_next
            v_true = graph.v_next
            mae_pos = mae(x_true, x_new)
            mae_vel = mae(v_true, v_new)
            mae_stress = mae(von_mises_true, von_mises)

            error_x.append(mae_pos)
            error_v.append(mae_vel)
            error_stress.append(mae_stress)

            # Prepare input graph for the next time step
            input_graph = graph

            boundary_condition = graph.x[:, 0]  # 1 fixed, 0 free
            node_force = graph.node_force_next  # external force on node is known

            categorical_node_features = boundary_condition.unsqueeze(1)
            valued_node_features = torch.concat([x_new, v_new, u, node_force], dim=1)

            src, dst = graph.edge_index
            xij = x_new[dst] - x_new[src]
            xij_norm = torch.norm(xij, dim=1).unsqueeze(1)
            uij = u[dst] - u[src]
            uij_norm = torch.norm(uij, dim=1).unsqueeze(1)

            edge_features = torch.concat([xij, xij_norm, uij, uij_norm], dim=1)

            _edge_attr = normalize(edge_features, edge_stats)
            _x = torch.concat(
                [
                    categorical_node_features,
                    normalize(valued_node_features, node_stats),
                ],
                dim=1,
            )
            _x[-1] = graph.x[
                -1
            ]  # Keep the force node features from the future unchanges
            input_graph.x = _x
            input_graph.edge_attr = _edge_attr
            input_graph.x_pos_t = x_new
            input_graph.x_vel_t = v_new
            input_graph.node_force = node_force

            # Store rollouts for visualization
            true_rollout.append(
                Data(
                    x_pos_t=x_true,
                    x_vel_t=v_true,
                    von_mises=von_mises_true,
                    x_element_connectivity=graph.x_element_connectivity,
                )
            )
            pred_rollout.append(
                Data(
                    x_pos_t=x_new,
                    x_vel_t=v_new,
                    von_mises=von_mises,
                    x_element_connectivity=graph.x_element_connectivity,
                )
            )
    return error_x, error_v, error_stress, true_rollout, pred_rollout
