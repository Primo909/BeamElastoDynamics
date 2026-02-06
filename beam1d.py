"""1D beam first-mode vibration with 3D wireframe animation.

Examples:
    python beam1d.py
    python beam1d.py --a 0.1 --b 0.08 --omega 4.0
    python beam1d.py --save-gif beam_mode1.gif --no-show
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def mode_shape(x, L):
    """First bending mode shape: phi(x) = sin(pi * x / L)."""
    return np.sin(np.pi * x / L)


def compute_displacement(x, t, L, a, b, omega):
    """Compute displacement at each node for a given time.

    ux(x,t) = a * phi(x) * sin(omega * t)
    uy(x,t) = b * phi(x) * sin(omega * t)
    uz(x,t) = 0
    """
    phi = (x**2) / L
    # mode_shape(x, L)
    s = np.sin(omega * t)
    ux = a * phi * s
    uy = b * phi * s
    uz = np.zeros_like(x)
    return ux, uy, uz


def build_cross_section_corners(W, D):
    """Build the 4 corners of the rectangular cross-section in the y-z plane.

    Returns corners in order: bottom-left, bottom-right, top-right, top-left.
    """
    hw, hd = W / 2, D / 2
    return np.array(
        [
            [-hw, -hd],
            [hw, -hd],
            [hw, hd],
            [-hw, hd],
        ]
    )


def compute_local_frames(cx, cy, cz):
    """Compute local coordinate frames along the deformed centerline.

    At each node, returns:
    - T: unit tangent along the beam
    - N: unit normal perpendicular to tangent (in x-y plane)
    - B: unit binormal (T x N complement)

    Cross-sections are placed in the N-B plane so their normal aligns with T.
    """
    P = np.column_stack([cx, cy, cz])  # (N, 3)

    # Tangent via finite differences
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]  # central
    T[0] = P[1] - P[0]  # forward
    T[-1] = P[-1] - P[-2]  # backward

    # Normalize tangents
    T_len = np.linalg.norm(T, axis=1, keepdims=True)
    T = T / T_len

    # Reference up direction (z-axis)
    ref = np.array([0.0, 0.0, 1.0])

    # N = ref x T  (perpendicular to tangent, lies in x-y plane)
    N_vec = np.cross(ref, T)
    N_len = np.linalg.norm(N_vec, axis=1, keepdims=True)
    N_vec = N_vec / N_len

    # B = T x N  (completes the right-handed frame, close to z-up)
    B_vec = np.cross(T, N_vec)

    return T, N_vec, B_vec


def build_beam_segments(x, ux, uy, uz, corners):
    """Build Line3DCollection segments for the wireframe beam.

    Cross-sections are rotated so their normal aligns with the local beam tangent.

    Wireframe consists of:
    - 4 longitudinal edges along the beam: 4 * (N-1) segments
    - N rectangular cross-section outlines: 4 * N segments
    Total: 8*N - 4 segments, shape (M, 2, 3).
    """
    cx, cy, cz = x + ux, uy.copy(), uz.copy()
    _, N_vec, B_vec = compute_local_frames(cx, cy, cz)

    center = np.column_stack([cx, cy, cz])  # (N_nodes, 3)
    N_nodes = len(x)

    # Compute 3D corner positions at each node in the local frame:
    # P_corner = P_center + dy * N + dz * B
    corner_pos = np.zeros((4, N_nodes, 3))
    for c in range(4):
        dy, dz = corners[c]
        corner_pos[c] = center + dy * N_vec + dz * B_vec

    # Longitudinal edges: for each corner, pair node i with node i+1
    long_segs = []
    for c in range(4):
        starts = corner_pos[c, :-1]
        ends = corner_pos[c, 1:]
        long_segs.append(np.stack([starts, ends], axis=1))

    # Cross-section outlines: at each node, 4 edges of the rectangle
    cross_segs = []
    for c in range(4):
        c_next = (c + 1) % 4
        starts = corner_pos[c]
        ends = corner_pos[c_next]
        cross_segs.append(np.stack([starts, ends], axis=1))

    return np.concatenate(long_segs + cross_segs, axis=0)


def build_centerline(x, ux, uy, uz):
    """Compute the deformed centerline coordinates."""
    return x + ux, uy.copy(), uz.copy()


def setup_figure(L, W, D, a, b):
    """Create the 3D figure and initial artists for animation."""
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")

    margin = 0.05
    ax.set_xlim(-a - margin, L + a + margin)
    ax.set_ylim(-W / 2 - b - margin, W / 2 + b + margin)
    ax.set_zlim(-D / 2 - margin, D / 2 + margin)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("1D Beam — First Mode Vibration")
    ax.set_box_aspect((L * 0.1, 0.8 * W, 2.5 * D))
    ax.view_init(elev=25, azim=-60)

    return fig, ax


def animate(
    frame,
    x,
    corners,
    L,
    a,
    b,
    omega,
    dt,
    wireframe_collection,
    centerline_artist,
    trail_artist,
    trail_history,
    time_text,
):
    """Update function for FuncAnimation."""
    t = frame * dt
    ux, uy, uz = compute_displacement(x, t, L, a, b, omega)

    # Update wireframe
    segments = build_beam_segments(x, ux, uy, uz, corners)
    wireframe_collection.set_segments(segments)

    # Update centerline
    cx, cy, cz = build_centerline(x, ux, uy, uz)
    centerline_artist.set_data(cx, cy)
    centerline_artist.set_3d_properties(cz)

    # Update trail (center point at x = L/2)
    mid = len(x) // 2
    trail_history["x"].append(cx[mid])
    trail_history["y"].append(cy[mid])
    trail_history["z"].append(cz[mid])
    trail_artist.set_data(trail_history["x"], trail_history["y"])
    trail_artist.set_3d_properties(trail_history["z"])

    time_text.set_text(f"t = {t:.3f} s")

    return wireframe_collection, centerline_artist, trail_artist, time_text


def main():
    parser = argparse.ArgumentParser(
        description="1D beam first-mode vibration with 3D wireframe animation."
    )
    parser.add_argument(
        "--L", type=float, default=1.0, help="Beam length (default: 1.0)"
    )
    parser.add_argument(
        "--W", type=float, default=0.1, help="Cross-section width (default: 0.1)"
    )
    parser.add_argument(
        "--D", type=float, default=0.1, help="Cross-section depth (default: 0.1)"
    )
    parser.add_argument(
        "--N", type=int, default=25, help="Number of nodes along beam (default: 25)"
    )
    parser.add_argument(
        "--a", type=float, default=0.05, help="Longitudinal amplitude (default: 0.05)"
    )
    parser.add_argument(
        "--b", type=float, default=0.03, help="Transverse amplitude (default: 0.03)"
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=2 * np.pi,
        help="Angular frequency (default: 2*pi)",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Animation duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--save-gif", type=str, default=None, help="Save animation as GIF to this path"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display animation interactively",
    )

    args = parser.parse_args()

    # Beam nodes along x-axis
    x = np.linspace(0, args.L, args.N)
    corners = build_cross_section_corners(args.W, args.D)

    # Animation timing
    n_frames = int(args.fps * args.duration)
    dt = args.duration / n_frames

    # Setup figure
    fig, ax = setup_figure(args.L, args.W, args.D, args.a, args.b)

    # Initial geometry at t=0
    ux, uy, uz = compute_displacement(x, 0.0, args.L, args.a, args.b, args.omega)
    segments = build_beam_segments(x, ux, uy, uz, corners)
    cx, cy, cz = build_centerline(x, ux, uy, uz)

    # Create artists
    wireframe_collection = Line3DCollection(
        segments, colors="steelblue", linewidths=0.8
    )
    ax.add_collection3d(wireframe_collection)

    (centerline_artist,) = ax.plot(cx, cy, cz, "r-", linewidth=2, label="Centerline")
    (trail_artist,) = ax.plot([], [], [], "g--", linewidth=1, alpha=0.6, label="Trail")
    trail_history = {"x": [], "y": [], "z": []}

    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    anim = FuncAnimation(
        fig,
        animate,
        frames=n_frames,
        fargs=(
            x,
            corners,
            args.L,
            args.a,
            args.b,
            args.omega,
            dt,
            wireframe_collection,
            centerline_artist,
            trail_artist,
            trail_history,
            time_text,
        ),
        interval=1000 / args.fps,
        blit=False,
    )

    if args.save_gif:
        anim.save(args.save_gif, writer="pillow", fps=args.fps)
        print(f"GIF saved => {args.save_gif}")

    if not args.no_show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
