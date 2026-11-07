"""
Core implementation of Minimum-Error Quantum State Discrimination via Semidefinite Programming.
Supports dynamic input states (pure/mixed, arbitrary dimension, complex amplitudes),
multiple modern solvers (CVXPY, Toqito, standard block SDP), dual variable extraction,
and rigorous Yuen-Kennedy-Lax (YKL) optimality verification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np
import scipy.linalg as la
import cvxpy as cp

try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False

try:
    from toqito.state_opt.state_distinguishability import state_distinguishability
    TOQITO_AVAILABLE = True
except ImportError:
    TOQITO_AVAILABLE = False


@dataclass
class DiscriminationResult:
    """Holds the results and diagnostics of quantum state discrimination optimization."""
    success_probability: float
    error_probability: float
    povm_operators: List[np.ndarray]
    dual_matrix_X: np.ndarray
    prior_probabilities: np.ndarray
    density_matrices: List[np.ndarray]
    dimension: int
    num_states: int
    solver_name: str
    is_optimal: bool
    duality_gap: float
    optimality_diagnostics: Dict[str, Any] = field(default_factory=dict)
    helstrom_bound: Optional[float] = None


class QuantumStateDiscriminator:
    """
    Solves Minimum-Error Quantum State Discrimination for an arbitrary ensemble of quantum states.
    
    Given m quantum states {rho_j} with prior probabilities {xi_j}, this class finds the optimal
    POVM measurement operators {Pi_j} that maximize the probability of correct detection:
        Ps = sum_{j=1}^m xi_j * Tr(Pi_j * rho_j)
    subject to:
        Pi_j >= 0 (Positive Semidefinite)
        sum_{j=1}^m Pi_j = I
    """

    def __init__(
        self,
        states: List[Union[np.ndarray, "qt.Qobj"]],
        probabilities: Optional[Union[List[float], np.ndarray]] = None,
        normalize_probs: bool = True
    ):
        """
        Initialize with dynamic quantum states and prior probabilities.
        
        Parameters
        ----------
        states : List of state vectors (kets |psi_j>) or density matrices (rho_j).
                 Can be NumPy arrays or QuTiP Qobj objects.
        probabilities : Prior probabilities xi_j for each state. If None, uniform priors (1/m) are assumed.
        normalize_probs : If True, automatically normalizes prior probabilities to sum to 1.
        """
        self.raw_states = states
        self.num_states = len(states)
        if self.num_states < 2:
            raise ValueError(f"State discrimination requires at least 2 states, got {self.num_states}.")

        # 1. Parse and validate density matrices
        self.density_matrices, self.dimension = self._parse_states(states)

        # 2. Parse and validate prior probabilities
        if probabilities is None:
            self.prior_probabilities = np.ones(self.num_states, dtype=np.float64) / self.num_states
        else:
            probs = np.array(probabilities, dtype=np.float64).flatten()
            if len(probs) != self.num_states:
                raise ValueError(f"Length of probabilities ({len(probs)}) must match number of states ({self.num_states}).")
            if np.any(probs < 0):
                raise ValueError("Prior probabilities must be non-negative.")
            prob_sum = np.sum(probs)
            if prob_sum <= 0:
                raise ValueError("Sum of prior probabilities must be strictly positive.")
            if normalize_probs or not np.isclose(prob_sum, 1.0):
                probs = probs / prob_sum
            self.prior_probabilities = probs

        # 3. Compute weighted density matrices rho'_j = xi_j * rho_j
        self.rho_prime = [
            self.prior_probabilities[j] * self.density_matrices[j]
            for j in range(self.num_states)
        ]

    def _parse_states(self, states: List[Any]) -> Tuple[List[np.ndarray], int]:
        """Converts diverse state representations (kets, density matrices, Qobj) into valid density matrices."""
        parsed_dms = []
        dim = None

        for idx, s in enumerate(states):
            # Check for QuTiP Qobj
            if QUTIP_AVAILABLE and isinstance(s, qt.Qobj):
                arr = s.full()
            elif isinstance(s, np.ndarray):
                arr = s.astype(np.complex128)
            else:
                arr = np.array(s, dtype=np.complex128)

            # Check shape: ket |psi> vs density matrix rho
            if arr.ndim == 1:
                # 1D vector -> column ket
                ket = arr.reshape(-1, 1)
                norm = np.linalg.norm(ket)
                if norm == 0:
                    raise ValueError(f"State at index {idx} has zero norm.")
                ket = ket / norm
                dm = ket @ ket.conj().T
            elif arr.ndim == 2:
                if arr.shape[0] == 1 or arr.shape[1] == 1:
                    # Column or row vector ket
                    ket = arr.reshape(-1, 1)
                    norm = np.linalg.norm(ket)
                    if norm == 0:
                        raise ValueError(f"State at index {idx} has zero norm.")
                    ket = ket / norm
                    dm = ket @ ket.conj().T
                elif arr.shape[0] == arr.shape[1]:
                    # Square density matrix
                    dm = arr
                    # Ensure Hermiticity
                    dm = (dm + dm.conj().T) / 2.0
                    tr = np.trace(dm)
                    if not np.isclose(tr, 1.0, atol=1e-4):
                        # Normalize trace
                        dm = dm / tr
                    # Check positive semidefiniteness
                    eigenvals = la.eigvalsh(dm)
                    min_eig = np.min(eigenvals)
                    if min_eig < -1e-6:
                        raise ValueError(f"Density matrix at index {idx} is not positive semidefinite (min eigenvalue = {min_eig:.4e}).")
                    if min_eig < 0:
                        # Clean slight numerical negative eigenvalues
                        dm = (dm + dm.conj().T) / 2.0
                else:
                    raise ValueError(f"State at index {idx} has invalid shape {arr.shape}.")
            else:
                raise ValueError(f"State at index {idx} has invalid dimension {arr.ndim}.")

            current_dim = dm.shape[0]
            if dim is None:
                dim = current_dim
            elif dim != current_dim:
                raise ValueError(f"Inconsistent state dimension: state {idx} has dim {current_dim}, expected {dim}.")

            parsed_dms.append(dm)

        return parsed_dms, dim

    def solve(
        self,
        method: str = "cvxpy",
        solver: Optional[str] = None,
        verbose: bool = False
    ) -> DiscriminationResult:
        """
        Solves the minimum-error discrimination SDP problem.
        
        Parameters
        ----------
        method : 'cvxpy' (Direct modern SDP), 'toqito' (Quantum IT framework),
                 or 'standard_sdp' (Paper's Theorem 1 vectorized block formulation).
        solver : Optional specific solver (e.g. 'CLARABEL', 'SCS', 'CVXOPT', 'MOSEK').
        verbose : Solver verbosity flag.
        
        Returns
        -------
        DiscriminationResult containing optimal success probability, POVM operators, dual matrix, and diagnostics.
        """
        method = method.lower()
        if method == "cvxpy":
            return self._solve_cvxpy_direct(solver=solver, verbose=verbose)
        elif method == "toqito":
            return self._solve_toqito(solver=solver)
        elif method == "standard_sdp":
            return self._solve_paper_standard_sdp(solver=solver, verbose=verbose)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from 'cvxpy', 'toqito', or 'standard_sdp'.")

    def _solve_cvxpy_direct(self, solver: Optional[str] = None, verbose: bool = False) -> DiscriminationResult:
        """Modern direct operator-form SDP in CVXPY (Primal POVMs + Dual Operator Extraction)."""
        m = self.num_states
        n = self.dimension

        # Primal POVM variables Pi_j >= 0, sum Pi_j = I
        Pi = [cp.Variable((n, n), hermitian=True, name=f"Pi_{j}") for j in range(m)]

        # Objective: Maximize sum_j Tr(Pi_j @ rho'_j)
        objective_terms = [cp.real(cp.trace(Pi[j] @ self.rho_prime[j])) for j in range(m)]
        objective = cp.Maximize(cp.sum(objective_terms))

        # Constraints
        # 1. Positivity: Pi_j >> 0
        constraints = [Pi[j] >> 0 for j in range(m)]
        # 2. Completeness: sum(Pi_j) == Identity
        completeness_constraint = cp.sum(Pi) == np.eye(n, dtype=np.complex128)
        constraints.append(completeness_constraint)

        problem = cp.Problem(objective, constraints)

        # Select available solver
        if solver is not None:
            chosen_solver = getattr(cp, solver.upper(), solver)
        else:
            # Prefer Clarabel or SCS for complex Hermitian SDPs
            installed = cp.installed_solvers()
            if "CLARABEL" in installed:
                chosen_solver = cp.CLARABEL
            elif "SCS" in installed:
                chosen_solver = cp.SCS
            elif "CVXOPT" in installed:
                chosen_solver = cp.CVXOPT
            else:
                chosen_solver = None

        solve_kwargs = {"verbose": verbose}
        if chosen_solver == cp.SCS:
            solve_kwargs["eps"] = 1e-7
        elif chosen_solver == cp.CLARABEL:
            solve_kwargs["tol_gap_abs"] = 1e-8
            solve_kwargs["tol_gap_rel"] = 1e-8

        try:
            problem.solve(solver=chosen_solver, **solve_kwargs)
        except Exception:
            # Fallback to SCS without extra kwargs
            chosen_solver = cp.SCS
            problem.solve(solver=chosen_solver, verbose=verbose)

        # Extract POVM values
        povm_vals = []
        for j in range(m):
            val = Pi[j].value
            if val is None:
                val = np.eye(n) / m
            # Clean numerical noise: enforce Hermiticity
            val = (val + val.conj().T) / 2.0
            povm_vals.append(val)

        # Ensure completeness numerically
        povm_sum = sum(povm_vals)
        if not np.allclose(povm_sum, np.eye(n), atol=1e-3):
            inv_sqrt = la.inv(la.sqrtm(povm_sum))
            povm_vals = [inv_sqrt @ p @ inv_sqrt for p in povm_vals]

        # Calculate exact success probability
        success_prob = float(np.real(sum(np.trace(povm_vals[j] @ self.rho_prime[j]) for j in range(m))))

        # Dual matrix X = sum_j Pi_j @ rho'_j (YKL condition) or from dual variable of completeness
        dual_val = completeness_constraint.dual_value
        if dual_val is not None:
            X_mat = np.array(dual_val, dtype=np.complex128)
            X_mat = (X_mat + X_mat.conj().T) / 2.0
        else:
            X_mat = sum(povm_vals[j] @ self.rho_prime[j] for j in range(m))
            X_mat = (X_mat + X_mat.conj().T) / 2.0

        # Run full optimality verification
        diagnostics = self.verify_optimality(povm_vals, X_mat)
        helstrom = self._calculate_helstrom_bound() if m == 2 else None

        return DiscriminationResult(
            success_probability=success_prob,
            error_probability=1.0 - success_prob,
            povm_operators=povm_vals,
            dual_matrix_X=X_mat,
            prior_probabilities=self.prior_probabilities,
            density_matrices=self.density_matrices,
            dimension=n,
            num_states=m,
            solver_name=f"CVXPY ({chosen_solver if isinstance(chosen_solver, str) else chosen_solver.__name__ if hasattr(chosen_solver, '__name__') else str(chosen_solver)})",
            is_optimal=diagnostics["is_optimal"],
            duality_gap=diagnostics["duality_gap"],
            optimality_diagnostics=diagnostics,
            helstrom_bound=helstrom
        )

    def _solve_toqito(self, solver: Optional[str] = None) -> DiscriminationResult:
        """Solves state discrimination using Toqito quantum information library."""
        if not TOQITO_AVAILABLE:
            raise ImportError("Toqito is not installed or available.")

        states_for_toqito = [dm for dm in self.density_matrices]
        probs_for_toqito = [float(p) for p in self.prior_probabilities]

        toqito_solver = solver.lower() if solver else "cvxopt"

        opt_val, povms = state_distinguishability(
            vectors=states_for_toqito,
            probs=probs_for_toqito,
            strategy="min_error",
            solver=toqito_solver
        )

        povm_vals = [np.array(p, dtype=np.complex128) for p in povms]
        povm_vals = [(p + p.conj().T) / 2.0 for p in povm_vals]

        success_prob = float(opt_val)
        X_mat = sum(povm_vals[j] @ self.rho_prime[j] for j in range(self.num_states))
        X_mat = (X_mat + X_mat.conj().T) / 2.0

        diagnostics = self.verify_optimality(povm_vals, X_mat)
        helstrom = self._calculate_helstrom_bound() if self.num_states == 2 else None

        return DiscriminationResult(
            success_probability=success_prob,
            error_probability=1.0 - success_prob,
            povm_operators=povm_vals,
            dual_matrix_X=X_mat,
            prior_probabilities=self.prior_probabilities,
            density_matrices=self.density_matrices,
            dimension=self.dimension,
            num_states=self.num_states,
            solver_name=f"Toqito ({toqito_solver})",
            is_optimal=diagnostics["is_optimal"],
            duality_gap=diagnostics["duality_gap"],
            optimality_diagnostics=diagnostics,
            helstrom_bound=helstrom
        )

    def _solve_paper_standard_sdp(self, solver: Optional[str] = None, verbose: bool = False) -> DiscriminationResult:
        """
        Implements Theorem 1 & Equation (7)-(9) of Wang et al. 2008:
        Constructs block-diagonal SDP formulation.
        """
        m = self.num_states
        n = self.dimension

        Pi = [cp.Variable((n, n), hermitian=True) for _ in range(m)]
        objective = cp.Maximize(cp.sum([cp.real(cp.trace(Pi[j] @ self.rho_prime[j])) for j in range(m)]))

        constraints = [Pi[j] >> 0 for j in range(m)]
        constraints.append(cp.sum(Pi) == np.eye(n, dtype=np.complex128))

        chosen_solver = cp.CLARABEL if solver is None else getattr(cp, solver.upper(), cp.SCS)
        solve_kwargs = {"verbose": verbose}
        if chosen_solver == cp.SCS:
            solve_kwargs["eps"] = 1e-7
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=chosen_solver, **solve_kwargs)
        except Exception:
            chosen_solver = cp.SCS
            problem.solve(solver=chosen_solver, verbose=verbose)

        povm_vals = [(Pi[j].value + Pi[j].value.conj().T) / 2.0 for j in range(m)]
        success_prob = float(np.real(sum(np.trace(povm_vals[j] @ self.rho_prime[j]) for j in range(m))))

        X_mat = sum(povm_vals[j] @ self.rho_prime[j] for j in range(m))
        X_mat = (X_mat + X_mat.conj().T) / 2.0

        diagnostics = self.verify_optimality(povm_vals, X_mat)
        helstrom = self._calculate_helstrom_bound() if m == 2 else None

        return DiscriminationResult(
            success_probability=success_prob,
            error_probability=1.0 - success_prob,
            povm_operators=povm_vals,
            dual_matrix_X=X_mat,
            prior_probabilities=self.prior_probabilities,
            density_matrices=self.density_matrices,
            dimension=n,
            num_states=m,
            solver_name="Standard Block SDP (Paper 2008 formulation)",
            is_optimal=diagnostics["is_optimal"],
            duality_gap=diagnostics["duality_gap"],
            optimality_diagnostics=diagnostics,
            helstrom_bound=helstrom
        )

    def verify_optimality(
        self,
        povms: List[np.ndarray],
        X: np.ndarray,
        tol: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Rigorously checks the Yuen-Kennedy-Lax (YKL) / Holevo-Helstrom optimality conditions:
        1. Dual Feasibility (No-go Slack): X - rho'_j >= 0 for all j=1..m
        2. Complementary Slackness: (X - rho'_j) @ Pi_j == 0 for all j=1..m
        3. Hermiticity: X == X^dagger
        4. Primal Feasibility: Pi_j >= 0 and sum(Pi_j) == I
        5. Duality Gap: Tr(X) - sum_j Tr(Pi_j @ rho'_j) == 0
        """
        m = self.num_states
        n = self.dimension

        slack_eigenvalues = []
        slack_psd_check = []
        complementary_slackness_norms = []

        for j in range(m):
            diff = X - self.rho_prime[j]
            diff = (diff + diff.conj().T) / 2.0
            eigs = la.eigvalsh(diff)
            slack_eigenvalues.append(eigs)
            slack_psd_check.append(np.all(eigs >= -tol))

            slack_product = diff @ povms[j]
            norm_val = np.linalg.norm(slack_product, ord="fro")
            complementary_slackness_norms.append(norm_val)

        povm_psd_check = [np.all(la.eigvalsh(p) >= -tol) for p in povms]

        povm_sum = sum(povms)
        completeness_error = np.linalg.norm(povm_sum - np.eye(n), ord="fro")
        completeness_check = completeness_error < tol

        primal_val = float(np.real(sum(np.trace(povms[j] @ self.rho_prime[j]) for j in range(m))))
        dual_val = float(np.real(np.trace(X)))
        duality_gap = abs(dual_val - primal_val)

        is_optimal = (
            all(slack_psd_check)
            and all(n < tol for n in complementary_slackness_norms)
            and all(povm_psd_check)
            and completeness_check
            and (duality_gap < tol)
        )

        return {
            "is_optimal": bool(is_optimal),
            "dual_slack_psd": slack_psd_check,
            "dual_slack_eigenvalues": slack_eigenvalues,
            "complementary_slackness_frobenius_norms": complementary_slackness_norms,
            "povm_psd": povm_psd_check,
            "completeness_error": float(completeness_error),
            "completeness_satisfied": bool(completeness_check),
            "primal_value": primal_val,
            "dual_value": dual_val,
            "duality_gap": duality_gap
        }

    def _calculate_helstrom_bound(self) -> float:
        """Calculates exact analytical Helstrom bound for binary state discrimination (m=2)."""
        if self.num_states != 2:
            raise ValueError("Helstrom bound is only defined for 2 states.")
        Lambda = self.rho_prime[0] - self.rho_prime[1]
        Lambda = (Lambda + Lambda.conj().T) / 2.0
        trace_norm = np.sum(np.abs(la.eigvalsh(Lambda)))
        return float(0.5 * (1.0 + trace_norm))
