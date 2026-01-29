import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing_extensions import Literal

from mesh_utils import volume
from preprocess_data import Stats, denormalize, normalize


def mae(target, pred):
    return torch.mean(torch.abs(target - pred))


def project_tetrahedron_volume(
    x: torch.Tensor,
    v: torch.Tensor,
    tet: torch.Tensor,
    V_rest: float,
    dt: float,
    fixed_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project a single tetrahedron to preserve its rest volume.

    Args:
        x: Node positions (N, 3)
        v: Node velocities (N, 3)
        tet: Tetrahedron indices (4,)
        V_rest: Rest volume of the tetrahedron
        dt: Time step
        fixed_mask: Boolean mask indicating fixed nodes (N,)

    Returns:
        Updated (x, v)
    """
    # Get the four vertices
    indices = tet.long()
    x_tet = x[indices]  # (4, 3)

    # Check if any vertices are fixed
    is_fixed = fixed_mask[indices]

    # If all vertices are fixed, no correction needed
    if is_fixed.all():
        return x, v

    # Calculate current volume
    a, b, c, d = x_tet[0], x_tet[1], x_tet[2], x_tet[3]
    current_vol = torch.abs(
        -torch.sum((a - d) * torch.cross(b - c, c - d, dim=0)) / 6.0
    )

    # Volume error
    V_error = current_vol - V_rest

    # If volume is already close enough, skip correction
    if torch.abs(V_error) < 1e-8 * V_rest:
        return x, v

    # Compute volume gradient with respect to each vertex
    # For a tetrahedron with vertices (a, b, c, d), the volume gradient is:
    # dV/da = (1/6) * (c-b) x (d-b)
    # dV/db = (1/6) * (c-d) x (a-d)
    # dV/dc = (1/6) * (d-b) x (a-b)
    # dV/dd = (1/6) * (b-c) x (a-c)

    grad_a = torch.cross(c - b, d - b, dim=0) / 6.0
    grad_b = torch.cross(c - d, a - d, dim=0) / 6.0
    grad_c = torch.cross(d - b, a - b, dim=0) / 6.0
    grad_d = torch.cross(b - c, a - c, dim=0) / 6.0

    # Handle sign based on current volume calculation
    if torch.sum((a - d) * torch.cross(b - c, c - d, dim=0)) < 0:
        grad_a = -grad_a
        grad_b = -grad_b
        grad_c = -grad_c
        grad_d = -grad_d

    grads = torch.stack([grad_a, grad_b, grad_c, grad_d], dim=0)  # (4, 3)

    # Zero out gradients for fixed nodes
    grads[is_fixed] = 0.0

    # Compute denominator (squared norm of gradients)
    denom = torch.sum(grads * grads)

    if denom < 1e-12:
        # No free nodes to move
        return x, v

    # Compute correction factor (Lagrange multiplier)
    lambda_corr = -V_error / denom

    # Update positions
    dx = lambda_corr * grads
    x_new = x.clone()
    x_new[indices] = x_new[indices] + dx

    # Update velocities to maintain consistency
    v_new = v.clone()
    if dt > 1e-12:
        dv = dx / dt
        v_new[indices] = v_new[indices] + dv

    return x_new, v_new


def apply_volume_preservation(
    x: torch.Tensor,
    v: torch.Tensor,
    cells: torch.Tensor,
    V_rest: torch.Tensor,
    dt: float,
    fixed_mask: torch.Tensor,
    n_iters: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply volume preservation constraints using Gauss-Seidel iterations.

    Args:
        x: Node positions (N, 3)
        v: Node velocities (N, 3)
        cells: Tetrahedron connectivity (M, 4)
        V_rest: Rest volumes for each tetrahedron (M,)
        dt: Time step
        fixed_mask: Boolean mask indicating fixed nodes (N,)
        n_iters: Number of Gauss-Seidel iterations

    Returns:
        Updated (x, v) with volume constraints enforced
    """
    for _ in range(n_iters):
        for tet_idx in range(cells.shape[0]):
            tet = cells[tet_idx]
            V_rest_elem = V_rest[tet_idx]
            x, v = project_tetrahedron_volume(x, v, tet, V_rest_elem, dt, fixed_mask)

    return x, v


def build_mgn_graph_from_timestep(
    simulation_graph: Data,
    x_new: torch.Tensor,
    v_new: torch.Tensor,
    u: torch.Tensor,
    node_stats: Stats,
    edge_stats: Stats,
    target_stats: Stats,
):
    input_graph = simulation_graph

    boundary_condition = simulation_graph.x[:, 0]  # 1 fixed, 0 free
    node_force = simulation_graph.node_force_next  # external force on node is known

    categorical_node_features = boundary_condition.unsqueeze(1)
    valued_node_features = torch.concat([x_new, v_new, u, node_force], dim=1)

    src, dst = simulation_graph.edge_index
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
    _x[-1] = simulation_graph.x[
        -1
    ]  # Keep the force node features from the future unchanges
    input_graph.x = _x
    input_graph.edge_attr = _edge_attr
    input_graph.x_pos_t = x_new
    input_graph.x_vel_t = v_new
    input_graph.node_force = node_force
    return input_graph


def do_rollout(
    model: torch.nn.Module,
    test_loader: DataLoader,
    node_stats: Stats,
    edge_stats: Stats,
    target_stats: Stats,
    dt: float,
    device: torch.device,
    skip_first: int = 0,
    rollout_steps: int = 99999999,
    dont_rollout: bool = False,
    integrator: Literal[
        "semiimplicit_euler", "explicit_euler", "trapezoidal"
    ] = "semiimplicit_euler",
    preserve_volume: bool = False,
    volume_reference: Literal["rest", "last_step"] = "last_step",
    volume_iters: int = 10,
):
    """
    Returns `true_rollout, pred_rollout`

    Args:
        preserve_volume: Whether to apply volume preservation constraints
        volume_reference: 'rest' preserves volume w.r.t. initial rest state,
                         'last_step' preserves volume w.r.t. previous timestep
        volume_iters: Number of Gauss-Seidel iterations for volume projection
    """
    with torch.no_grad():
        # Store errors for evaluation
        error_stress = []
        true_rollout = []
        pred_rollout = []

        # Rest volumes (computed once from the first graph if needed)
        V_rest = None
        cells = None
        u = None

        for t, graph in enumerate(test_loader):
            if u is None:
                u = graph.u
            if t < skip_first:
                continue
            if t > rollout_steps:
                break
            if t == skip_first:
                input_graph = graph
                # Compute rest volumes from the initial configuration
                if preserve_volume and volume_reference == "rest":
                    cells = torch.tensor(
                        graph.x_element_connectivity[0], dtype=torch.long
                    )
                    cells = cells.squeeze()
                    print("u", u.shape)
                    print("cells", cells.shape)
                    V_rest = torch.abs(volume(u, cells))
            if dont_rollout:
                input_graph = graph

            # Initialize cells if not already done
            if preserve_volume and cells is None:
                cells = torch.tensor(
                    graph.x_element_connectivity[0], dtype=torch.long
                ).squeeze()

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

            # Compute volumes from current state if preserving w.r.t. last step
            if preserve_volume and volume_reference == "last_step":
                V_rest = torch.abs(volume(x, cells))

            # integrate
            boundary_condition = graph.x[:, 0]  # 1 fixed, 0 free

            v_new = v + a * dt
            x_new = x + v_new * dt

            # Apply volume preservation before enforcing boundary conditions
            if preserve_volume and V_rest is not None:
                fixed_mask = boundary_condition == 1
                x_new, v_new = apply_volume_preservation(
                    x_new, v_new, cells, V_rest, dt, fixed_mask, n_iters=volume_iters
                )

            # Enforce fixed boundary conditions
            x_new[boundary_condition == 1] = x[boundary_condition == 1]
            v_new[boundary_condition == 1] = 0.0

            # Use new graph to calculate error
            x_true = graph.x_next
            v_true = graph.v_next

            # Prepare input graph for the next time step
            input_graph = build_mgn_graph_from_timestep(
                simulation_graph=graph,
                x_new=x_new,
                v_new=v_new,
                u=u,
                node_stats=node_stats,
                edge_stats=edge_stats,
                target_stats=target_stats,
            )

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
    return true_rollout, pred_rollout
