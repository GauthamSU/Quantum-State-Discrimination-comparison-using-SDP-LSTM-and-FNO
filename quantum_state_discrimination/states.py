"""
Dynamic quantum state generators and standard quantum information benchmarks.
Supports pure states (kets), mixed states (density matrices), arbitrary dimensions (qubits, qutrits, qudits),
and standard benchmark ensembles (Paper 2008, Trine, BB84, SIC/Tetrahedron, Mirror-Symmetric, Haar-Random).
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import scipy.linalg as la


def paper_2008_states() -> Tuple[List[np.ndarray], List[float]]:
    """
    Returns the exact 3-state benchmark ensemble from Section 5 of Wang et al. 2008:
        |phi_1> = [1, 0]^T
        |phi_2> = 1/sqrt(2) * [1, 1]^T
        |phi_3> = [0, 1]^T
        Prior probabilities: xi = [0.1, 0.6, 0.3]
    """
    phi1 = np.array([1.0, 0.0], dtype=np.complex128)
    phi2 = (1.0 / np.sqrt(2.0)) * np.array([1.0, 1.0], dtype=np.complex128)
    phi3 = np.array([0.0, 1.0], dtype=np.complex128)

    states = [phi1, phi2, phi3]
    probs = [0.1, 0.6, 0.3]
    return states, probs


def trine_states(theta_offset: float = 0.0) -> Tuple[List[np.ndarray], List[float]]:
    """
    Returns 3 symmetric Trine states in 2D separated by 120 degrees (2*pi/3) in the X-Z plane:
        |psi_k> = cos(theta_k) |0> + sin(theta_k) |1>
        where theta_k = theta_offset + 2*pi*k / 3, for k = 0, 1, 2.
    Default prior probabilities are uniform [1/3, 1/3, 1/3].
    """
    states = []
    for k in range(3):
        angle = theta_offset + 2.0 * np.pi * k / 3.0
        ket = np.array([np.cos(angle), np.sin(angle)], dtype=np.complex128)
        states.append(ket)
    probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    return states, probs


def bb84_states() -> Tuple[List[np.ndarray], List[float]]:
    """
    Returns the 4 standard BB84 polarization/qubit states:
        |0>, |1>, |+> = (|0>+|1>)/sqrt(2), |-> = (|0>-|1>)/sqrt(2)
    Prior probabilities: uniform [0.25, 0.25, 0.25, 0.25].
    """
    s0 = np.array([1.0, 0.0], dtype=np.complex128)
    s1 = np.array([0.0, 1.0], dtype=np.complex128)
    sp = (1.0 / np.sqrt(2.0)) * np.array([1.0, 1.0], dtype=np.complex128)
    sm = (1.0 / np.sqrt(2.0)) * np.array([1.0, -1.0], dtype=np.complex128)

    states = [s0, s1, sp, sm]
    probs = [0.25, 0.25, 0.25, 0.25]
    return states, probs


def tetrahedron_states() -> Tuple[List[np.ndarray], List[float]]:
    """
    Returns 4 qubit states whose Bloch vectors form a regular tetrahedron (SIC-POVM in d=2):
        |v_0> = |0>
        |v_1> = sqrt(1/3)|0> + sqrt(2/3)|1>
        |v_2> = sqrt(1/3)|0> + sqrt(2/3) e^(i 2pi/3) |1>
        |v_3> = sqrt(1/3)|0> + sqrt(2/3) e^(i 4pi/3) |1>
    """
    v0 = np.array([1.0, 0.0], dtype=np.complex128)
    v1 = np.array([np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0)], dtype=np.complex128)
    v2 = np.array([np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0) * np.exp(1j * 2.0 * np.pi / 3.0)], dtype=np.complex128)
    v3 = np.array([np.sqrt(1.0 / 3.0), np.sqrt(2.0 / 3.0) * np.exp(1j * 4.0 * np.pi / 3.0)], dtype=np.complex128)

    states = [v0, v1, v2, v3]
    probs = [0.25, 0.25, 0.25, 0.25]
    return states, probs


def mirror_symmetric_states(theta: float = np.pi / 4.0, probs: Optional[List[float]] = None) -> Tuple[List[np.ndarray], List[float]]:
    """
    Returns 3 mirror-symmetric qubit states (Andersson et al. 2002):
        |psi_1> = cos(theta)|0> + sin(theta)|1>
        |psi_2> = |0>
        |psi_3> = cos(theta)|0> - sin(theta)|1>
    """
    psi1 = np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128)
    psi2 = np.array([1.0, 0.0], dtype=np.complex128)
    psi3 = np.array([np.cos(theta), -np.sin(theta)], dtype=np.complex128)

    states = [psi1, psi2, psi3]
    if probs is None:
        probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    return states, probs


def qubit_from_bloch(theta: float, phi: float) -> np.ndarray:
    """
    Creates a pure qubit state vector from spherical Bloch coordinates:
        |psi(theta, phi)> = cos(theta/2)|0> + e^(i*phi) sin(theta/2)|1>
    """
    return np.array([
        np.cos(theta / 2.0),
        np.exp(1j * phi) * np.sin(theta / 2.0)
    ], dtype=np.complex128)


def random_pure_states(
    num_states: int,
    dimension: int = 2,
    seed: Optional[int] = None
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Generates m Haar-random pure quantum states in Hilbert space dimension n.
    """
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(num_states):
        # Complex standard normal components
        real_part = rng.normal(size=dimension)
        imag_part = rng.normal(size=dimension)
        vec = real_part + 1j * imag_part
        norm = np.linalg.norm(vec)
        states.append(vec / norm)

    # Uniform prior probabilities
    probs = (np.ones(num_states) / num_states).tolist()
    return states, probs


def random_mixed_states(
    num_states: int,
    dimension: int = 2,
    rank: Optional[int] = None,
    seed: Optional[int] = None
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Generates m random mixed quantum states (density matrices) in Hilbert space dimension n
    sampled from the Hilbert-Schmidt ensemble.
    """
    rng = np.random.default_rng(seed)
    states = []
    r = dimension if rank is None else min(rank, dimension)

    for _ in range(num_states):
        # Ginibre matrix G of shape (dimension, r)
        G = rng.normal(size=(dimension, r)) + 1j * rng.normal(size=(dimension, r))
        # Wishart matrix W = G @ G^dagger (guaranteed positive semidefinite)
        W = G @ G.conj().T
        # Normalize trace
        rho = W / np.trace(W)
        states.append(rho)

    probs = (np.ones(num_states) / num_states).tolist()
    return states, probs


def add_depolarizing_noise(
    states: List[np.ndarray],
    noise_parameter_p: float
) -> List[np.ndarray]:
    """
    Applies depolarizing channel with noise parameter p in [0, 1] to a list of states:
        E(rho) = (1 - p) * rho + p * (I / d)
    """
    noisy_states = []
    for s in states:
        if s.ndim == 1 or (s.ndim == 2 and (s.shape[0] == 1 or s.shape[1] == 1)):
            ket = s.reshape(-1, 1)
            ket = ket / np.linalg.norm(ket)
            rho = ket @ ket.conj().T
        else:
            rho = s
        d = rho.shape[0]
        identity = np.eye(d, dtype=np.complex128) / d
        noisy_rho = (1.0 - noise_parameter_p) * rho + noise_parameter_p * identity
        noisy_states.append(noisy_rho)
    return noisy_states
