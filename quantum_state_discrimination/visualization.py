"""
Visualization and reporting utilities for Quantum State Discrimination.
Supports Bloch sphere rendering (QuTiP / Matplotlib 3D), optimality condition spectra,
parameter sweeps, and rich result summaries.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Any, Callable
import numpy as np
import matplotlib.pyplot as plt

try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False

from .core import DiscriminationResult


def density_to_bloch_vector(rho: np.ndarray) -> np.ndarray:
    """
    Computes Cartesian Bloch vector [rx, ry, rz] for a single-qubit density matrix rho:
        rx = Tr(rho * sigma_x)
        ry = Tr(rho * sigma_y)
        rz = Tr(rho * sigma_z)
    """
    if rho.shape != (2, 2):
        raise ValueError(f"Bloch vector is only defined for 2x2 qubit density matrices, got shape {rho.shape}.")
    
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    rx = float(np.real(np.trace(rho @ sigma_x)))
    ry = float(np.real(np.trace(rho @ sigma_y)))
    rz = float(np.real(np.trace(rho @ sigma_z)))
    return np.array([rx, ry, rz])


def plot_bloch_sphere(
    states: List[np.ndarray],
    povms: Optional[List[np.ndarray]] = None,
    state_labels: Optional[List[str]] = None,
    povm_labels: Optional[List[str]] = None,
    title: str = "Bloch Sphere: Quantum States and Optimal Detection POVMs",
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Renders states and optimal POVM measurement operators on the Bloch sphere.
    """
    # Verify qubit dimension
    for s in states:
        dim = s.shape[0] if s.ndim == 2 else len(s)
        if dim != 2:
            raise ValueError(f"Bloch sphere visualization requires 2D qubit states, got dimension {dim}.")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_wireframe(x, y, z, color='lightgray', alpha=0.35, linewidth=0.5)
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.08)

    # Draw axes
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k--', alpha=0.4, linewidth=0.8)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k--', alpha=0.4, linewidth=0.8)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k--', alpha=0.4, linewidth=0.8)

    ax.text(1.25, 0, 0, r'$|x+\rangle$', color='gray', fontsize=9)
    ax.text(0, 1.25, 0, r'$|y+\rangle$', color='gray', fontsize=9)
    ax.text(0, 0, 1.25, r'$|0\rangle$', color='gray', fontsize=9)
    ax.text(0, 0, -1.35, r'$|1\rangle$', color='gray', fontsize=9)

    # Plot Quantum States
    state_colors = plt.cm.Set1(np.linspace(0, 0.8, len(states)))
    for idx, s in enumerate(states):
        if s.ndim == 1 or (s.ndim == 2 and (s.shape[0] == 1 or s.shape[1] == 1)):
            ket = s.reshape(-1, 1)
            ket = ket / np.linalg.norm(ket)
            rho = ket @ ket.conj().T
        else:
            rho = s
        
        vec = density_to_bloch_vector(rho)
        lbl = state_labels[idx] if state_labels and idx < len(state_labels) else f"State $|\\psi_{idx+1}\\rangle$"
        color = state_colors[idx]

        # Draw vector arrow
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color, linewidth=2.5, arrow_length_ratio=0.1)
        ax.scatter([vec[0]], [vec[1]], [vec[2]], color=color, s=50, label=lbl)

    # Plot POVM measurement vectors if provided
    if povms is not None:
        povm_colors = plt.cm.Dark2(np.linspace(0, 0.8, len(povms)))
        for idx, p in enumerate(povms):
            trace_p = float(np.real(np.trace(p)))
            if trace_p > 1e-4:
                # Normalize POVM element to extract direction
                p_normed = p / trace_p
                vec = density_to_bloch_vector(p_normed)
                lbl = povm_labels[idx] if povm_labels and idx < len(povm_labels) else f"POVM $\\Pi_{idx+1}$ (Tr={trace_p:.2f})"
                color = povm_colors[idx]

                ax.quiver(0, 0, 0, vec[0] * trace_p, vec[1] * trace_p, vec[2] * trace_p,
                          color=color, linewidth=2.0, linestyle='--', arrow_length_ratio=0.12)
                ax.scatter([vec[0] * trace_p], [vec[1] * trace_p], [vec[2] * trace_p],
                           color=color, marker='^', s=60, label=lbl)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_zlim([-1.3, 1.3])
    ax.set_title(title, fontsize=12, pad=15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=9)
    plt.tight_layout()
    return fig


def plot_optimality_verification(result: DiscriminationResult, figsize: Tuple[int, int] = (10, 4)) -> plt.Figure:
    """
    Plots the eigenvalue spectrum of the dual slack matrices (X - rho'_j)
    to visually confirm dual feasibility (X - rho'_j >= 0) and complementary slackness.
    """
    m = result.num_states
    n = result.dimension
    eigenvals = result.optimality_diagnostics.get("dual_slack_eigenvalues", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 1. Dual Slack Eigenvalues (Must all be >= 0)
    x_indices = np.arange(1, m + 1)
    for eig_idx in range(n):
        vals = [eigenvals[j][eig_idx] for j in range(m)]
        ax1.plot(x_indices, vals, 'o-', linewidth=2, markersize=8, label=rf"$\lambda_{{{eig_idx+1}}}(X - \rho'_j)$")

    ax1.axhline(0, color='red', linestyle='--', linewidth=1.2, label=r'PSD Boundary ($\lambda = 0$)')
    ax1.set_xlabel(r"State Index $j$", fontsize=10)
    ax1.set_ylabel("Eigenvalue", fontsize=10)
    ax1.set_title(r"Dual Slack PSD Check ($X - \rho'_j \succeq 0$)", fontsize=11)
    ax1.set_xticks(x_indices)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # 2. Complementary Slackness Norms: ||(X - rho'_j) * Pi_j||_F (Must be ~ 0)
    slack_norms = result.optimality_diagnostics.get("complementary_slackness_frobenius_norms", [0] * m)
    bars = ax2.bar(x_indices, slack_norms, color='teal', alpha=0.7, edgecolor='black')
    ax2.set_xlabel(r"State Index $j$", fontsize=10)
    ax2.set_ylabel(r"Frobenius Norm $\|(X - \rho'_j)\Pi_j\|_F$", fontsize=10)
    ax2.set_title(r"Complementary Slackness ($\approx 0$)", fontsize=11)
    ax2.set_xticks(x_indices)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add numeric labels on bars
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1e-6, f"{yval:.2e}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    return fig


def plot_prior_probability_sweep(
    states: List[np.ndarray],
    prior_ratios: np.ndarray,
    target_idx: int = 1,
    title: str = "Optimal Success Probability vs Prior Probability",
    figsize: Tuple[int, int] = (8, 4.5)
) -> plt.Figure:
    """
    Sweeps the prior probability xi_k of a chosen state across [0.01, 0.99]
    while keeping remaining states equiprobable, and plots Ps vs xi_k.
    """
    from .core import QuantumStateDiscriminator

    m = len(states)
    success_probs = []

    for p_val in prior_ratios:
        rem_p = (1.0 - p_val) / (m - 1)
        probs = [p_val if i == target_idx else rem_p for i in range(m)]
        disc = QuantumStateDiscriminator(states=states, probabilities=probs)
        res = disc.solve(method="cvxpy")
        success_probs.append(res.success_probability)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(prior_ratios, success_probs, 'b-', linewidth=2.5, label=r"$P_s$ (SDP Optimal)")
    ax.plot(prior_ratios, prior_ratios, 'k--', linewidth=1, alpha=0.6, label=r"Trivial Guess ($\max \xi_j$)")

    ax.set_xlabel(f"Prior Probability $\\xi_{target_idx+1}$", fontsize=11)
    ax.set_ylabel("Detection Probability $P_s$", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig


def display_results_summary(result: DiscriminationResult) -> None:
    """
    Prints a clean, formatted report of the state discrimination optimization.
    """
    print("=" * 65)
    print(f"  QUANTUM STATE DISCRIMINATION RESULTS ({result.solver_name})")
    print("=" * 65)
    print(f"Hilbert Space Dimension (n):       {result.dimension}")
    print(f"Number of Quantum States (m):      {result.num_states}")
    print(f"Optimal Success Probability (Ps):   {result.success_probability:.6f} ({result.success_probability*100:.2f}%)")
    print(f"Minimum Error Probability (Pe):     {result.error_probability:.6f} ({result.error_probability*100:.2f}%)")
    if result.helstrom_bound is not None:
        print(f"Helstrom Analytical Bound (m=2):    {result.helstrom_bound:.6f}")
    print(f"Duality Gap:                       {result.duality_gap:.2e}")
    print(f"Optimality Verification (YKL):     {'PASSED (Globally Optimal)' if result.is_optimal else 'FAILED'}")
    print("-" * 65)
    print("Prior Probabilities xi:")
    print(" ", [round(float(p), 4) for p in result.prior_probabilities])
    print("-" * 65)
    print("Optimal POVM Measurement Operators {Pi_j}:")
    for j, p in enumerate(result.povm_operators):
        print(f"\nPOVM Pi_{j+1} (Trace = {np.real(np.trace(p)):.4f}):")
        print(np.round(p, 4))
    print("=" * 65)
