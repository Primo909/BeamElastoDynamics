import torch


def volume(V: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    # Dimension:
    a = V[T[:, 0], :]
    b = V[T[:, 1], :]
    c = V[T[:, 2], :]
    d = V[T[:, 3], :]
    vols = -torch.sum(torch.mul(a - d, torch.cross(b - c, c - d, dim=1)), dim=1) / 6.0
    return vols

def batched_tetrahedron_volumes(x: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """
    Calculates volumes matching the user's specific logic and output format.
    
    Args:
        x: (B, N, 3)
        cells: (B, M, 4)
        
    Returns:
        volumes: (B*M)  <-- Flattened to match your loop's output
    """
    B, M, _ = cells.shape

    # 1. Create Batch Indices for gathering
    # We need to grab specific nodes from specific batches
    # batch_idx: (B, M, 4)
    batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, M, 4)
    
    # 2. Gather coordinates
    # x[batch_idx, cells] allows us to pick node 'cells[b,m,k]' from batch 'b'
    # Result: (B, M, 4, 3)
    tetra_coords = x[batch_idx, cells]

    # 3. Unbind into vertices (A, B, C, D)
    # Each shape: (B, M, 3)
    a, b, c, d = tetra_coords.unbind(dim=2)

    # 4. Compute vectors exactly as in your loop
    # Your logic: cross(b-c, c-d) dot (a-d)
    vec_a_d = a - d
    vec_b_c = b - c
    vec_c_d = c - d

    # 5. Cross Product (b-c) x (c-d)
    # dim=-1 ensures we cross along the coordinate dimension (x,y,z)
    cross_prod = torch.cross(vec_b_c, vec_c_d, dim=-1)

    # 6. Dot Product (manual)
    # sum(vec_a_d * cross, dim=-1)
    scalar_triple = torch.sum(vec_a_d * cross_prod, dim=-1)

    # 7. Volume and Flatten
    volumes = torch.abs(scalar_triple) / 6.0
    
    # Flatten to (B*M) to match your 'all_volumes' list structure
    return volumes


if __name__ == "__main__":
    # Simple test
    v = torch.tensor([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ]
    )
    # This tet is properly oriented
    t = torch.tensor([[[0, 1, 2, 3],[0, 1, 2, 3]],])
    vols = batched_tetrahedron_volumes(v, t)
    print("Volume should be 1/6:", vols)
