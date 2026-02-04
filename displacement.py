import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def create_rectangle_mesh(w, h, nx=20, ny=20):
    """Create a grid of points for a rectangle."""
    z1 = jnp.linspace(0, w, nx)
    z2 = jnp.linspace(0, h, ny)
    Z1, Z2 = jnp.meshgrid(z1, z2)
    return Z1, Z2


def displacement_u1(z1, z2, w, h, a, b):
    """Compute u1 component of displacement."""
    return (z1 - w / 2) * a


def displacement_u2(z1, z2, w, h, a, b):
    """Compute u2 component of displacement."""
    c = w * h / (2 * (1 + a)) * 1.3
    print(f"c = {c}")
    return ((z1 - w / 2) ** 2 * b - c) * (z2 - h / 2)


def displacement(Z1, Z2, w, h, a, b):
    """Compute displacement field u(z)."""
    u1 = displacement_u1(Z1, Z2, w, h, a, b)
    u2 = displacement_u2(Z1, Z2, w, h, a, b)
    return u1, u2


def deform(Z1, Z2, u1, u2):
    """Apply deformation: x = z + u."""
    X1 = Z1 + u1
    X2 = Z2 + u2
    return X1, X2


def compute_strain_eij(Z1, Z2, w, h, a, b, i, j):
    """Compute strain component eij = 1/2 * (dui/dzj + duj/dzi) using JAX autodiff.

    Args:
        Z1, Z2: Mesh coordinates
        w, h, a, b: Deformation parameters
        i, j: Strain component indices (1 or 2)

    Returns:
        eij: Strain component at each mesh point
    """
    # Map indices to displacement functions
    u_funcs = {1: displacement_u1, 2: displacement_u2}

    # Gradient of ui with respect to zj (argnums: 0=z1, 1=z2)
    dui_dzj = jax.grad(u_funcs[i], argnums=j - 1)
    duj_dzi = jax.grad(u_funcs[j], argnums=i - 1)

    # Vectorize over the mesh
    def compute_eij_point(z1, z2):
        return 0.5 * (dui_dzj(z1, z2, w, h, a, b) + duj_dzi(z1, z2, w, h, a, b))

    compute_eij = jax.vmap(jax.vmap(compute_eij_point))
    eij = compute_eij(Z1, Z2)
    return eij


def compute_stress_sigma_ij(Z1, Z2, w, h, a, b, lam, mu, i, j):
    """Compute stress component sigma_ij = lambda * tr(e) * delta_ij + 2 * mu * e_ij.

    Args:
        Z1, Z2: Mesh coordinates
        w, h, a, b: Deformation parameters
        lam: First Lamé parameter (lambda)
        mu: Second Lamé parameter (shear modulus)
        i, j: Stress component indices (1 or 2)

    Returns:
        sigma_ij: Stress component at each mesh point
    """
    # Compute strain components
    e11 = compute_strain_eij(Z1, Z2, w, h, a, b, 1, 1)
    e22 = compute_strain_eij(Z1, Z2, w, h, a, b, 2, 2)
    eij = compute_strain_eij(Z1, Z2, w, h, a, b, i, j)

    # Trace of strain tensor
    trace_e = e11 + e22

    # Kronecker delta
    delta_ij = 1.0 if i == j else 0.0

    # Stress: sigma_ij = lambda * tr(e) * delta_ij + 2 * mu * e_ij
    sigma_ij = lam * trace_e * delta_ij + 2 * mu * eij

    return sigma_ij


def compute_von_mises_stress(Z1, Z2, w, h, a, b, lam, mu):
    """Compute von Mises stress for 2D plane stress.

    von Mises stress: σ_vm = sqrt(σ₁₁² - σ₁₁σ₂₂ + σ₂₂² + 3σ₁₂²)

    Args:
        Z1, Z2: Mesh coordinates
        w, h, a, b: Deformation parameters
        lam: First Lamé parameter (lambda)
        mu: Second Lamé parameter (shear modulus)

    Returns:
        sigma_vm: Von Mises stress at each mesh point
    """
    sigma_11 = compute_stress_sigma_ij(Z1, Z2, w, h, a, b, lam, mu, 1, 1)
    sigma_22 = compute_stress_sigma_ij(Z1, Z2, w, h, a, b, lam, mu, 2, 2)
    sigma_12 = compute_stress_sigma_ij(Z1, Z2, w, h, a, b, lam, mu, 1, 2)

    sigma_vm = jnp.sqrt(
        sigma_11**2 - sigma_11 * sigma_22 + sigma_22**2 + 3 * sigma_12**2
    )
    return sigma_vm


def compute_stress_divergence(Z1, Z2, w, h, a, b, lam, mu):
    """Compute divergence of stress tensor: div(σ)_i = ∂σ_ij/∂z_j.

    In 2D:
        div(σ)_1 = ∂σ₁₁/∂z₁ + ∂σ₁₂/∂z₂
        div(σ)_2 = ∂σ₂₁/∂z₁ + ∂σ₂₂/∂z₂

    Args:
        Z1, Z2: Mesh coordinates
        w, h, a, b: Deformation parameters
        lam: First Lamé parameter (lambda)
        mu: Second Lamé parameter (shear modulus)

    Returns:
        div_sigma_1, div_sigma_2: Divergence components at each mesh point
    """
    u_funcs = {1: displacement_u1, 2: displacement_u2}

    def sigma_ij_point(z1, z2, i, j):
        """Compute stress component sigma_ij at a single point."""

        # Compute strain components at this point
        def eij_point(i, j):
            dui_dzj = jax.grad(u_funcs[i], argnums=j - 1)
            duj_dzi = jax.grad(u_funcs[j], argnums=i - 1)
            return 0.5 * (dui_dzj(z1, z2, w, h, a, b) + duj_dzi(z1, z2, w, h, a, b))

        e11 = eij_point(1, 1)
        e22 = eij_point(2, 2)
        eij = eij_point(i, j)
        trace_e = e11 + e22
        delta_ij = 1.0 if i == j else 0.0
        return lam * trace_e * delta_ij + 2 * mu * eij

    def div_sigma_point(z1, z2):
        """Compute divergence of stress at a single point."""
        # div(σ)_1 = ∂σ₁₁/∂z₁ + ∂σ₁₂/∂z₂
        dsigma11_dz1 = jax.grad(lambda z1, z2: sigma_ij_point(z1, z2, 1, 1), argnums=0)(
            z1, z2
        )
        dsigma12_dz2 = jax.grad(lambda z1, z2: sigma_ij_point(z1, z2, 1, 2), argnums=1)(
            z1, z2
        )
        div_1 = dsigma11_dz1 + dsigma12_dz2

        # div(σ)_2 = ∂σ₂₁/∂z₁ + ∂σ₂₂/∂z₂
        dsigma21_dz1 = jax.grad(lambda z1, z2: sigma_ij_point(z1, z2, 2, 1), argnums=0)(
            z1, z2
        )
        dsigma22_dz2 = jax.grad(lambda z1, z2: sigma_ij_point(z1, z2, 2, 2), argnums=1)(
            z1, z2
        )
        div_2 = dsigma21_dz1 + dsigma22_dz2

        return div_1, div_2

    # Vectorize over the mesh
    compute_div = jax.vmap(jax.vmap(lambda z1, z2: div_sigma_point(z1, z2)))
    div_sigma_1, div_sigma_2 = compute_div(Z1, Z2)

    return div_sigma_1, div_sigma_2


def plot_configurations(
    w, h, a, b, lam=1.0, mu=1.0, strain_ij=(2, 2), stress_ij=(2, 2)
):
    """Plot original and deformed rectangle with strain and stress colorbars.

    Args:
        w, h: Rectangle dimensions
        a, b: Deformation parameters
        lam, mu: Lamé parameters
        strain_ij: Tuple (i, j) specifying which strain component to plot
        stress_ij: Tuple (i, j) specifying which stress component to plot
    """
    # Create mesh
    Z1, Z2 = create_rectangle_mesh(w, h)

    # Compute deformation
    u1, u2 = displacement(Z1, Z2, w, h, a, b)
    X1, X2 = deform(Z1, Z2, u1, u2)

    # Compute strain using JAX autodiff
    ei, ej = strain_ij
    strain = compute_strain_eij(Z1, Z2, w, h, a, b, i=ei, j=ej)

    # Compute stress
    si, sj = stress_ij
    stress = compute_stress_sigma_ij(Z1, Z2, w, h, a, b, lam, mu, i=si, j=sj)

    # Compute von Mises stress
    von_mises = compute_von_mises_stress(Z1, Z2, w, h, a, b, lam, mu)

    # Subscript labels
    subscripts = {1: "₁", 2: "₂"}
    strain_label = f"e{subscripts[ei]}{subscripts[ej]}"
    stress_label = f"σ{subscripts[si]}{subscripts[sj]}"

    # Create figure with 2x3 subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))

    # Plot original configuration
    ax1.plot(Z1, Z2, "b-", linewidth=0.5)
    ax1.plot(Z1.T, Z2.T, "b-", linewidth=0.5)
    ax1.set_xlabel("z₁")
    ax1.set_ylabel("z₂")
    ax1.set_title("Original Configuration")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Plot deformed configuration with displacement vector field
    ax2.plot(X1, X2, "r-", linewidth=0.5)
    ax2.plot(X1.T, X2.T, "r-", linewidth=0.5)
    # Draw displacement vectors from original to deformed positions
    ax2.quiver(
        np.array(Z1),
        np.array(Z2),
        np.array(u1),
        np.array(u2),
        angles="xy",
        scale_units="xy",
        scale=1,
        color="blue",
        alpha=0.7,
        width=0.005,
    )
    ax2.set_xlabel("x₁")
    ax2.set_ylabel("x₂")
    ax2.set_title("Deformed Configuration with Displacement Field")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    # Plot deformed configuration with strain colormap
    pcm1 = ax3.pcolormesh(
        np.array(X1),
        np.array(X2),
        np.array(strain),
        shading="auto",
        cmap="jet",
    )
    ax3.plot(X1, X2, "k-", linewidth=0.3, alpha=0.5)
    ax3.plot(X1.T, X2.T, "k-", linewidth=0.3, alpha=0.5)
    ax3.set_xlabel("x₁")
    ax3.set_ylabel("x₂")
    ax3.set_title(f"Strain {strain_label}")
    ax3.set_aspect("equal")
    cbar1 = fig.colorbar(pcm1, ax=ax3)
    cbar1.set_label(strain_label)

    # Plot deformed configuration with stress colormap
    pcm2 = ax4.pcolormesh(
        np.array(X1),
        np.array(X2),
        np.array(stress),
        shading="auto",
        cmap="jet",
    )
    ax4.plot(X1, X2, "k-", linewidth=0.3, alpha=0.5)
    ax4.plot(X1.T, X2.T, "k-", linewidth=0.3, alpha=0.5)
    ax4.set_xlabel("x₁")
    ax4.set_ylabel("x₂")
    ax4.set_title(f"Stress {stress_label}")
    ax4.set_aspect("equal")
    cbar2 = fig.colorbar(pcm2, ax=ax4)
    cbar2.set_label(stress_label)

    # Plot deformed configuration with von Mises stress colormap
    pcm3 = ax5.pcolormesh(
        np.array(X1),
        np.array(X2),
        np.array(von_mises),
        shading="auto",
        cmap="jet",
    )
    ax5.plot(X1, X2, "k-", linewidth=0.3, alpha=0.5)
    ax5.plot(X1.T, X2.T, "k-", linewidth=0.3, alpha=0.5)
    ax5.set_xlabel("x₁")
    ax5.set_ylabel("x₂")
    ax5.set_title("Von Mises Stress σ_vm")
    ax5.set_aspect("equal")
    cbar3 = fig.colorbar(pcm3, ax=ax5)
    cbar3.set_label("σ_vm")

    # Hide the unused 6th subplot
    ax6.axis("off")

    # Compute axis limits based on data extent
    all_x = np.concatenate([np.array(Z1).flatten(), np.array(X1).flatten()])
    all_y = np.concatenate([np.array(Z2).flatten(), np.array(X2).flatten()])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    x_margin = 0.1 * (x_max - x_min)
    y_margin = 0.1 * (y_max - y_min)

    # Set axis limits
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    plt.tight_layout()
    plt.savefig("displacement_plot.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # Default parameters
    w = 1.0  # width
    h = 0.5  # height
    a = 0.1  # deformation parameter
    b = 2 * h / (w * (1 + a))  # deformation parameter
    lam = 1.0  # First Lamé parameter
    mu = 1.0  # Second Lamé parameter (shear modulus)
    strain_ij = (2, 2)  # Strain component to plot (e.g., (1,1), (2,2), (1,2))
    stress_ij = (2, 2)  # Stress component to plot (e.g., (1,1), (2,2), (1,2))

    plot_configurations(w, h, a, b, lam, mu, strain_ij, stress_ij)

    # Compute and check divergence of stress tensor
    print("\n" + "=" * 60)
    print("Divergence of Stress Tensor Check")
    print("=" * 60)

    Z1, Z2 = create_rectangle_mesh(w, h)
    div_sigma_1, div_sigma_2 = compute_stress_divergence(Z1, Z2, w, h, a, b, lam, mu)

    print("div(σ)₁ = ∂σ₁₁/∂z₁ + ∂σ₁₂/∂z₂:")
    print(f"  Max absolute value: {jnp.max(jnp.abs(div_sigma_1)):.2e}")
    print(f"  Min value: {jnp.min(div_sigma_1):.2e}")
    print(f"  Max value: {jnp.max(div_sigma_1):.2e}")

    print("\ndiv(σ)₂ = ∂σ₂₁/∂z₁ + ∂σ₂₂/∂z₂:")
    print(f"  Max absolute value: {jnp.max(jnp.abs(div_sigma_2)):.2e}")
    print(f"  Min value: {jnp.min(div_sigma_2):.2e}")
    print(f"  Max value: {jnp.max(div_sigma_2):.2e}")

    tol = 1e-10
    is_zero_1 = jnp.allclose(div_sigma_1, 0.0, atol=tol)
    is_zero_2 = jnp.allclose(div_sigma_2, 0.0, atol=tol)

    print(f"\nEquilibrium check (tolerance={tol}):")
    print(f"  div(σ)₁ ≈ 0: {is_zero_1}")
    print(f"  div(σ)₂ ≈ 0: {is_zero_2}")

    if is_zero_1 and is_zero_2:
        print("\n✓ Stress tensor divergence is ZERO - equilibrium satisfied!")
    else:
        print(
            "\n✗ Stress tensor divergence is NON-ZERO - body forces required for equilibrium"
        )
