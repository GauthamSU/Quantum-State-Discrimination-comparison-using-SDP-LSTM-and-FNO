"""
Comprehensive Unit Tests for QuantumStateDiscriminator and Modern SDP State Discrimination.
"""

import numpy as np
import scipy.linalg as la
import pytest
from quantum_state_discrimination import QuantumStateDiscriminator, states
import qutip as qt


def test_paper_2008_benchmark():
    """Verifies exact reproduction of Section 5 in Wang et al. 2008."""
    raw_states, probs = states.paper_2008_states()
    discriminator = QuantumStateDiscriminator(states=raw_states, probabilities=probs)

    # 1. CVXPY direct solver
    res_cvxpy = discriminator.solve(method="cvxpy")
    expected_ps = 0.78541019652
    assert np.isclose(res_cvxpy.success_probability, expected_ps, atol=1e-4)
    assert res_cvxpy.is_optimal

    # Check POVM elements
    # Pi_1 should be zero
    assert np.allclose(res_cvxpy.povm_operators[0], 0.0, atol=1e-3)
    # Pi_2 and Pi_3 elements from paper
    expected_pi2 = np.array([[0.7236, 0.4472], [0.4472, 0.2764]])
    expected_pi3 = np.array([[0.2764, -0.4472], [-0.4472, 0.7236]])
    assert np.allclose(np.real(res_cvxpy.povm_operators[1]), expected_pi2, atol=1e-3)
    assert np.allclose(np.real(res_cvxpy.povm_operators[2]), expected_pi3, atol=1e-3)

    # 2. Standard block SDP solver (Theorem 1 formulation)
    res_std = discriminator.solve(method="standard_sdp")
    assert np.isclose(res_std.success_probability, expected_ps, atol=1e-4)
    assert res_std.is_optimal

    # 3. Toqito solver
    res_toqito = discriminator.solve(method="toqito")
    assert np.isclose(res_toqito.success_probability, expected_ps, atol=1e-4)
    assert res_toqito.is_optimal


def test_helstrom_bound_binary_states():
    """Verifies that 2-state discrimination matches exact Helstrom analytical bound."""
    theta = np.pi / 6.0
    # Two non-orthogonal pure states
    s1 = np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128)
    s2 = np.array([np.cos(theta), -np.sin(theta)], dtype=np.complex128)
    probs = [0.4, 0.6]

    discriminator = QuantumStateDiscriminator(states=[s1, s2], probabilities=probs)
    res = discriminator.solve(method="cvxpy")

    assert res.helstrom_bound is not None
    assert np.isclose(res.success_probability, res.helstrom_bound, atol=1e-5)
    assert res.is_optimal


def test_trine_states():
    """Verifies that 3 symmetric Trine states give Ps = 2/3."""
    trine, probs = states.trine_states()
    discriminator = QuantumStateDiscriminator(states=trine, probabilities=probs)
    res = discriminator.solve(method="cvxpy")

    # Analytical result for 3 symmetric trine states is 2/3
    assert np.isclose(res.success_probability, 2.0 / 3.0, atol=1e-4)
    assert res.is_optimal


def test_bb84_states():
    """Verifies that 4 BB84 states give Ps = 1/2."""
    bb84, probs = states.bb84_states()
    discriminator = QuantumStateDiscriminator(states=bb84, probabilities=probs)
    res = discriminator.solve(method="cvxpy")

    # For BB84 with equal priors, optimal success prob is 1/2
    assert np.isclose(res.success_probability, 0.5, atol=1e-4)
    assert res.is_optimal


def test_qutrit_and_qudit_dynamic_states():
    """Verifies support for higher dimensions (qutrits n=3, qudits n=4) and random states."""
    # 4 random pure qutrit states (n=3)
    rand_qutrits, probs_3 = states.random_pure_states(num_states=4, dimension=3, seed=42)
    disc_3 = QuantumStateDiscriminator(states=rand_qutrits, probabilities=probs_3)
    res_3 = disc_3.solve(method="cvxpy")
    assert res_3.is_optimal
    assert 0 < res_3.success_probability <= 1.0

    # 5 random mixed qudit states (n=4)
    rand_qudits, probs_4 = states.random_mixed_states(num_states=5, dimension=4, seed=42)
    disc_4 = QuantumStateDiscriminator(states=rand_qudits, probabilities=probs_4)
    res_4 = disc_4.solve(method="cvxpy")
    assert res_4.is_optimal
    assert 0 < res_4.success_probability <= 1.0


def test_qutip_qobj_input():
    """Verifies seamless compatibility with QuTiP Qobj state representations."""
    psi0 = qt.basis(2, 0)
    psi1 = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    discriminator = QuantumStateDiscriminator(states=[psi0, psi1], probabilities=[0.5, 0.5])
    res = discriminator.solve(method="cvxpy")
    assert res.is_optimal
    assert np.isclose(res.success_probability, res.helstrom_bound, atol=1e-5)


def test_ykl_optimality_conditions():
    """Rigorously checks that dual slack X - rho'_j >= 0 and (X - rho'_j)*Pi_j = 0."""
    raw_states, probs = states.random_pure_states(num_states=3, dimension=2, seed=123)
    disc = QuantumStateDiscriminator(states=raw_states, probabilities=probs)
    res = disc.solve(method="cvxpy")

    diag = res.optimality_diagnostics
    assert diag["is_optimal"]
    assert all(diag["dual_slack_psd"])
    assert all(norm < 1e-4 for norm in diag["complementary_slackness_frobenius_norms"])
    assert diag["duality_gap"] < 1e-4


if __name__ == "__main__":
    pytest.main(["-v", __file__])
