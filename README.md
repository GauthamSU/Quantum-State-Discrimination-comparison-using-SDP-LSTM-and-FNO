# Minimum-Error Quantum State Discrimination via Semidefinite Programming (SDP)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Optimization-CVXPY](https://img.shields.io/badge/Optimization-CVXPY%201.9+-success.svg)](https://www.cvxpy.org/)
[![Quantum-Toqito](https://img.shields.io/badge/Quantum-Toqito%201.3+-purple.svg)](https://toqito.readthedocs.io/)
[![Quantum-QuTiP](https://img.shields.io/badge/Quantum-QuTiP%205.3+-orange.svg)](https://qutip.org/)
[![Tests-Pytest](https://img.shields.io/badge/Tests-7%20Passed%20(100%25)-brightgreen.svg)](test_discriminator.py)
[![License-MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced, publication-grade implementation and generalization of **Minimum-Error Quantum State Discrimination (MED)** based on:

> **Wang Wengang, Dong Guohua, Li Mingshan**,  
> *"Minimum-Error Quantum State Discrimination Based on Semidefinite Programming"*,  
> **Proceedings of the 27th Chinese Control Conference (CCC 2008)**, July 16–18, 2008, Kunming, Yunnan, China (IEEE), pp. 521–524.

This repository modernizes the 2008 MATLAB/SeDuMi numerical formulation into a flexible, multi-solver Python 3 ecosystem. It extends the original qubit formulation to **arbitrary Hilbert space dimensions ($n \ge 2$: qubits, qutrits, qudits)**, **arbitrary ensemble sizes ($m \ge 2$)**, **pure or mixed density matrices**, **non-uniform prior distributions**, and provides **rigorous automated certification of the Yuen-Kennedy-Lax (YKL) / Holevo-Helstrom optimality conditions**, 3D Bloch sphere projections, and interactive Jupyter controls.

---

## Table of Contents

- [1. Executive Overview](#1-executive-overview)
- [2. Theoretical Foundations](#2-theoretical-foundations)
  - [2.1. The Quantum State Discrimination Problem](#21-the-quantum-state-discrimination-problem)
  - [2.2. Primal Semidefinite Program (SDP)](#22-primal-semidefinite-program-sdp)
  - [2.3. Dual SDP Formulation](#23-dual-sdp-formulation)
  - [2.4. Yuen-Kennedy-Lax (YKL) Optimality Conditions](#24-yuen-kennedy-lax-ykl-optimality-conditions)
  - [2.5. Theorem 1 Block SDP Formulation (Wang et al. 2008)](#25-theorem-1-block-sdp-formulation-wang-et-al-2008)
  - [2.6. Binary States and the Analytical Helstrom Bound](#26-binary-states-and-the-analytical-helstrom-bound)
- [3. Modern Enhancements & Architecture](#3-modern-enhancements--architecture)
  - [3.1. Tri-Solver Optimization Engine](#31-tri-solver-optimization-engine)
  - [3.2. Dynamic State Engine & Benchmark Ensembles](#32-dynamic-state-engine--benchmark-ensembles)
  - [3.3. Rigorous Automated Optimality Certification](#33-rigorous-automated-optimality-certification)
  - [3.4. 3D Bloch Sphere & Parameter Visualizations](#34-3d-bloch-sphere--parameter-visualizations)
- [4. Exact Paper Reproduction (Section 5 Benchmark)](#4-exact-paper-reproduction-section-5-benchmark)
- [5. Project Structure](#5-project-structure)
- [6. Installation & Environment Setup](#6-installation--environment-setup)
- [7. Quickstart Guide & Code Examples](#7-quickstart-guide--code-examples)
  - [7.1. Solving the Paper Benchmark](#71-solving-the-paper-benchmark)
  - [7.2. Multi-Solver Cross-Verification](#72-multi-solver-cross-verification)
  - [7.3. Binary Discrimination vs. Analytical Helstrom Bound](#73-binary-discrimination-vs-analytical-helstrom-bound)
  - [7.4. Higher Dimensions: Qutrits ($n=3$) and Qudits ($n=4$)](#74-higher-dimensions-qutrits-n3-and-qudits-n4)
  - [7.5. Quantum Channel Decoherence (Depolarizing Noise)](#75-quantum-channel-decoherence-depolarizing-noise)
  - [7.6. QuTiP `Qobj` Interoperability](#76-qutip-qobj-interoperability)
- [8. Running the Unit Test Suite](#8-running-the-unit-test-suite)
- [9. Interactive Jupyter Notebook (`main.ipynb`)](#9-interactive-jupyter-notebook-mainipynb)
- [10. References](#10-references)

---

## 1. Executive Overview

In quantum communication and quantum cryptography, information is encoded into quantum states. When a sender (Alice) transmits one of $m$ possible non-orthogonal quantum states $\{\rho_j\}$ with prior probabilities $\{\xi_j\}$, quantum mechanics strictly prohibits distinguishing them with certainty (due to the no-cloning theorem and Heisenberg uncertainty).

Two primary discrimination paradigms exist:
1. **Unambiguous State Discrimination (USD)**: Errors are strictly forbidden, but inconclusive outcomes ("I don't know") are permitted.
2. **Minimum-Error State Discrimination (MED)**: Every measurement produces a definite conclusive hypothesis, and the measurement operators are optimized to minimize the average probability of error (equivalently, maximize the probability of correct detection $P_s$).

Finding analytical closed-form detection operators for MED is notoriously difficult and known only for specialized symmetric cases. In 2008, Wang, Dong, and Li showed that MED can be formulated as a standard Semidefinite Program (SDP) and solved numerically using interior-point solvers (specifically MATLAB 7.0 and SeDuMi 1.05).

**What This Project Delivers**:
- **Modernization**: Replaces legacy 2008 MATLAB scripts with modern Python 3, leveraging **CVXPY 1.9+** with native complex Hermitian cone constraints (`>> 0`), **Clarabel**, **SCS**, **CVXOPT**, and the **Toqito 1.3+** quantum theory framework.
- **Full Generalization**: No hardcoded 2D pure states. Arbitrary Hilbert space dimensions $n$, arbitrary hypothesis counts $m$, mixed density matrices $\rho_j$, and arbitrary prior distributions $\xi_j$ are supported out of the box.
- **Tri-Solver Verification**: Direct CVXPY Operator SDP, Quantum Theory Toolkit (Toqito), and Wang et al. Theorem 1 Block SDP.
- **Mathematical Certification**: Automatic algebraic check of all Holevo-Helstrom and Yuen-Kennedy-Lax (YKL) optimality conditions.
- **Rich Visualizations**: 3D Bloch sphere projections of states and POVM operators, dual slack eigenvalue spectrum plots, complementary slackness diagnostics, and interactive IPyWidgets dashboards.

---

## 2. Theoretical Foundations

### 2.1. The Quantum State Discrimination Problem

Let $\mathcal{H}_n$ denote an $n$-dimensional complex Hilbert space. Alice prepares a quantum system in state $\rho_j \in \mathcal{D}(\mathcal{H}_n)$ with prior probability $\xi_j > 0$, where $\sum_{j=1}^m \xi_j = 1$.

Bob performs a generalized quantum measurement defined by a **Positive Operator-Valued Measure (POVM)**, which is a collection of $m$ Hermitian operators $\{\Pi_j\}_{j=1}^m$ acting on $\mathcal{H}_n$ satisfying:
$$\Pi_j \succeq 0 \quad \forall j \in \{1, \dots, m\} \quad (\text{Positivity})$$
$$\sum_{j=1}^m \Pi_j = I_n \quad (\text{Completeness})$$

When state $\rho_j$ was transmitted, the conditional probability that Bob detects state $k$ is given by Born's rule:
$$P(k \mid j) = \mathrm{Tr}(\Pi_k \rho_j)$$

The average probability of successful detection $P_s$ is:
$$P_s = \sum_{j=1}^m \xi_j P(j \mid j) = \sum_{j=1}^m \xi_j \mathrm{Tr}(\Pi_j \rho_j) = \sum_{j=1}^m \mathrm{Tr}(\Pi_j \rho'_j)$$
where $\rho'_j \equiv \xi_j \rho_j$ are the **unnormalized (weighted) density operators**.

### 2.2. Primal Semidefinite Program (SDP)

The objective of Minimum-Error Discrimination is to maximize $P_s$ over all valid POVMs:

$$\begin{aligned}
\text{maximize} \quad & \sum_{j=1}^m \mathrm{Tr}(\Pi_j \rho'_j) \\
\text{subject to} \quad & \sum_{j=1}^m \Pi_j = I_n \\
& \Pi_j \succeq 0, \quad j = 1, \dots, m
\end{aligned}$$

Because the objective function is linear in $\{\Pi_j\}$ and the constraints consist of affine equalities and positive semidefinite cone constraints, this is a convex optimization problem (specifically, an SDP). Consequently, **any local optimum is guaranteed to be a global optimum**.

### 2.3. Dual SDP Formulation

By constructing the Lagrangian:
$$\mathcal{L}(\{\Pi_j\}, X, \{Y_j\}) = \sum_{j=1}^m \mathrm{Tr}(\Pi_j \rho'_j) + \mathrm{Tr}\left(X \left(I_n - \sum_{j=1}^m \Pi_j\right)\right) + \sum_{j=1}^m \mathrm{Tr}(Y_j \Pi_j)$$
where $X = X^\dagger$ is the Hermitian dual multiplier for the completeness constraint and $Y_j \succeq 0$ are the dual slack multipliers for the positivity constraints.

Rearranging terms:
$$\mathcal{L} = \mathrm{Tr}(X) + \sum_{j=1}^m \mathrm{Tr}\left(\Pi_j (\rho'_j - X + Y_j)\right)$$

For the Lagrangian to be bounded from above over $\Pi_j \succeq 0$, we must have $Y_j = X - \rho'_j \succeq 0$. This yields the **Dual SDP**:

$$\begin{aligned}
\text{minimize} \quad & \mathrm{Tr}(X) \\
\text{subject to} \quad & X - \rho'_j \succeq 0, \quad j = 1, \dots, m \\
& X = X^\dagger \in \mathbb{C}^{n \times n}
\end{aligned}$$

By Slater's condition, strong duality holds strictly with **zero duality gap**:
$$P_s^* = \sum_{j=1}^m \mathrm{Tr}(\Pi_j^* \rho'_j) = \mathrm{Tr}(X^*)$$

### 2.4. Yuen-Kennedy-Lax (YKL) Optimality Conditions

From the Karush-Kuhn-Tucker (KKT) conditions and the pioneering work of Yuen, Kennedy, and Lax (1975) and Holevo (1973), a measurement strategy $\{\Pi_j\}$ is globally optimal if and only if there exists a Hermitian operator $X$ such that:

1. **Dual Feasibility (No-Go Slack)**:
   $$X - \rho'_j \succeq 0 \quad \forall j \in \{1, \dots, m\}$$
   *(All eigenvalues of each slack matrix $S_j = X - \rho'_j$ must be non-negative: $\lambda_{\min}(S_j) \ge 0$.)*

2. **Complementary Slackness**:
   $$(X - \rho'_j)\Pi_j = 0 \quad \forall j \in \{1, \dots, m\}$$
   *(The POVM element $\Pi_j$ can only have support on the null space of the dual slack $X - \rho'_j$.)*

3. **Hermitian Bound Operator**:
   Summing over $j$ and using $\sum \Pi_j = I_n$:
   $$X = \sum_{j=1}^m \Pi_j \rho'_j = \sum_{j=1}^m \rho'_j \Pi_j$$
   *(Thus $X$ is explicitly constructed from the primal solution and must be Hermitian: $X = X^\dagger$.)*

### 2.5. Theorem 1 Block SDP Formulation (Wang et al. 2008)

Wang et al. (2008) vectorized and block-diagonalized the problem to interface with classical standard-form SDP solvers (e.g., SeDuMi in MATLAB):

$$\begin{aligned}
\text{DSDP:} \quad & \max \quad -\mathrm{Tr}(F_0 Z) \\
\text{subject to} \quad & \mathrm{Tr}(F_i Z) = c_i, \quad i = 1, \dots, n^2 \\
& Z \succeq 0
\end{aligned}$$

where:
- $Z = \mathrm{diag}(\Pi_1, \Pi_2, \dots, \Pi_m) \in \mathbb{C}^{mn \times mn}$
- $F_0 = \mathrm{diag}(-\rho'_1, -\rho'_2, \dots, -\rho'_m) \in \mathbb{C}^{mn \times mn}$
- $c_i = \delta_{lk}$ for $i = l + (k-1)n$, $l, k \in \{1, \dots, n\}$
- $F_i = \frac{1}{2} \sum_{j=1}^m \left( |e_{l+(j-1)n}\rangle \langle e_{k+(j-1)n}| + |e_{k+(j-1)n}\rangle \langle e_{l+(j-1)n}| \right)$

Our package implements both this exact block-diagonal formulation (`method="standard_sdp"`) and the modern operator-form formulation (`method="cvxpy"`).

### 2.6. Binary States and the Analytical Helstrom Bound

For binary hypothesis testing ($m=2$), Helstrom (1976) proved that the optimal success probability has an exact closed-form expression:
$$\Lambda = \rho'_1 - \rho'_2 = \xi_1 \rho_1 - \xi_2 \rho_2$$
$$P_s^{\text{Helstrom}} = \frac{1}{2} \left(1 + \|\Lambda\|_1\right) = \frac{1}{2} \left(1 + \sum_{k} |\lambda_k(\Lambda)|\right)$$

The optimal detection operators are projectors onto the positive and negative eigenspaces of $\Lambda$:
$$\Pi_1 = \sum_{\lambda_k > 0} |v_k\rangle \langle v_k|, \quad \Pi_2 = I - \Pi_1$$

Our package automatically computes the analytical Helstrom bound whenever $m=2$ and verifies it against the SDP numerical solution.

---

## 3. Modern Enhancements & Architecture

### 3.1. Tri-Solver Optimization Engine

This implementation provides three distinct, decoupled solution pathways:

```
                              QuantumStateDiscriminator.solve()
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         │                                    │                                    │
         ▼                                    ▼                                    ▼
   method="cvxpy"                      method="toqito"                    method="standard_sdp"
(Modern Operator SDP)             (Quantum IT Framework)              (Theorem 1 Block Formulation)
 Native Complex Hermitian         State Distinguishability            Direct block-diagonalization
   PSD cone constraints            Specialized Quantum SDP              from Wang et al. (2008)
  (Clarabel / SCS / CVXOPT)         (CVXOPT interior-point)              (Exact historical match)
```

1. **`method="cvxpy"` (Recommended)**: Solves the SDP directly over matrix variables $\Pi_j$ with Hermitian cone constraints (`Pi[j] >> 0`). Uses Clarabel (default) or SCS with high precision tolerances ($\epsilon \le 10^{-8}$).
2. **`method="toqito"`**: Interfaces directly with the Theory of Quantum Information Toolkit (`toqito.state_opt.state_distinguishability`), providing external cross-validation.
3. **`method="standard_sdp"`**: Exact numerical reproduction of Theorem 1 from the 2008 paper, assembling the block-diagonal matrices $Z$, $F_0$, and $\{F_i\}$.

### 3.2. Dynamic State Engine & Benchmark Ensembles

The package removes all hardcoding and includes a rich suite of analytical and random state generators in [`quantum_state_discrimination/states.py`](quantum_state_discrimination/states.py):

| State Generator | Function | Description | Theoretical $P_s$ |
|---|---|---|---|
| **Wang et al. (2008)** | `states.paper_2008_states()` | 3 pure qubit states with priors $[0.1, 0.6, 0.3]$ | $\approx 0.785410$ |
| **Trine States** | `states.trine_states()` | 3 symmetric planar states separated by $120^\circ$ | Exactly $2/3 \approx 0.666667$ |
| **BB84 States** | `states.bb84_states()` | 4 states from 2 mutually unbiased bases $(\{|0\rangle, |1\rangle, |+\rangle, |-\rangle\})$ | Exactly $1/2 = 0.500000$ |
| **SIC-POVM / Tetrahedron** | `states.tetrahedron_states()` | 4 qubit states forming a regular tetrahedron on the Bloch sphere | $\approx 0.500000$ (with equal priors) |
| **Mirror-Symmetric** | `states.mirror_symmetric_states(theta)` | 3 states with reflection symmetry (Andersson et al. 2002) | Closed-form dependent on $\theta$ |
| **Haar-Random Pure States** | `states.random_pure_states(m, n)` | $m$ Haar-random pure kets in dimension $n$ (qubits, qutrits, qudits) | Dynamic |
| **Random Mixed States** | `states.random_mixed_states(m, n)` | $m$ random density matrices from the Hilbert-Schmidt ensemble | Dynamic |
| **Depolarizing Noise** | `states.add_depolarizing_noise(states, p)` | Models quantum channel decoherence: $\mathcal{E}(\rho) = (1-p)\rho + p \frac{I}{d}$ | Decreases smoothly to $1/m$ as $p \to 1$ |

### 3.3. Rigorous Automated Optimality Certification

Every call to `.solve()` executes `verify_optimality()`, which performs full algebraic validation:
- **Dual Slack PSD**: Checks that all eigenvalues of $S_j = X - \rho'_j$ satisfy $\lambda_k(S_j) \ge -\epsilon$ (where $\epsilon = 10^{-4}$).
- **Complementary Slackness**: Evaluates the Frobenius norm $\|(X - \rho'_j)\Pi_j\|_F < \epsilon$ for every $j \in \{1, \dots, m\}$.
- **POVM Positivity**: Checks $\lambda_k(\Pi_j) \ge -\epsilon$.
- **POVM Completeness**: Measures $\|\sum_{j=1}^m \Pi_j - I_n\|_F < \epsilon$.
- **Duality Gap**: Confirms $|\mathrm{Tr}(X) - P_s| < \epsilon$.

### 3.4. 3D Bloch Sphere & Parameter Visualizations

Located in [`quantum_state_discrimination/visualization.py`](quantum_state_discrimination/visualization.py):
- **3D Bloch Sphere**: Projects arbitrary single-qubit states $|\psi_j\rangle$ (or density matrices $\rho_j$) and optimal POVM vectors $\vec{r}_j$ onto the unit sphere via Pauli decomposition:
  $$r_x = \mathrm{Tr}(\rho \sigma_x), \quad r_y = \mathrm{Tr}(\rho \sigma_y), \quad r_z = \mathrm{Tr}(\rho \sigma_z)$$
- **Dual Slack & Slackness Spectra**: Dual eigenvalue bar charts and complementary slackness Frobenius norm plots demonstrating KKT convergence.
- **Prior Probability Sweeps**: Visualizes the smooth transition in detection strategy and $P_s$ as prior probabilities $\xi_j$ vary across $[0.01, 0.99]$.
- **Depolarizing Noise Decay Curves**: Plots detection degradation as environmental noise parameter $p \in [0, 1]$ increases.

---

## 4. Exact Paper Reproduction (Section 5 Benchmark)

Section 5 of Wang et al. (2008) presents a concrete 3-state discrimination problem in dimension $n=2$:

$$|\phi_1\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |\phi_2\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}, \quad |\phi_3\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

with non-uniform prior probabilities:
$$\xi = [0.1, 0.6, 0.3]$$

### Benchmark Numerical Results Comparison

| Parameter / Metric | Wang et al. 2008 (MATLAB 7 / SeDuMi 1.05) | Our Modern CVXPY Solver | Our Toqito Solver | Our Standard SDP (Theorem 1) |
|---|---|---|---|---|
| **Optimal Success Prob ($P_s$)** | **$0.78541019652$** | **$0.78541020$** | **$0.78541020$** | **$0.78541020$** |
| **Minimum Error Prob ($P_e$)** | $0.21458980348$ | $0.21458980$ | $0.21458980$ | $0.21458980$ |
| **$\Pi_1$ (Operator for $|\phi_1\rangle$)** | $\begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$ | $\begin{pmatrix} 0.0000 & 0.0000 \\ 0.0000 & 0.0000 \end{pmatrix}$ | $\begin{pmatrix} 0.0000 & 0.0000 \\ 0.0000 & 0.0000 \end{pmatrix}$ | $\begin{pmatrix} 0.0000 & 0.0000 \\ 0.0000 & 0.0000 \end{pmatrix}$ |
| **$\Pi_2$ (Operator for $|\phi_2\rangle$)** | $\begin{pmatrix} 0.7236 & 0.4472 \\ 0.4472 & 0.2764 \end{pmatrix}$ | $\begin{pmatrix} 0.7236 & 0.4472 \\ 0.4472 & 0.2764 \end{pmatrix}$ | $\begin{pmatrix} 0.7236 & 0.4472 \\ 0.4472 & 0.2764 \end{pmatrix}$ | $\begin{pmatrix} 0.7236 & 0.4472 \\ 0.4472 & 0.2764 \end{pmatrix}$ |
| **$\Pi_3$ (Operator for $|\phi_3\rangle$)** | $\begin{pmatrix} 0.2764 & -0.4472 \\ -0.4472 & 0.7236 \end{pmatrix}$ | $\begin{pmatrix} 0.2764 & -0.4472 \\ -0.4472 & 0.7236 \end{pmatrix}$ | $\begin{pmatrix} 0.2764 & -0.4472 \\ -0.4472 & 0.7236 \end{pmatrix}$ | $\begin{pmatrix} 0.2764 & -0.4472 \\ -0.4472 & 0.7236 \end{pmatrix}$ |
| **Duality Gap** | $0$ (Analytical) | $< 10^{-8}$ | $< 10^{-8}$ | $< 10^{-8}$ |
| **YKL Optimality Verified** | Yes (Manual) | **PASSED (Automated)** | **PASSED (Automated)** | **PASSED (Automated)** |

> **Key Observation**: Even though $|\phi_1\rangle$ is present in the ensemble with prior $\xi_1 = 0.1$, the optimal strategy sets $\Pi_1 = 0$! Bob achieves maximum global success by never guessing state 1, instead partitioning the Hilbert space entirely between states 2 and 3.

---

## 5. Project Structure

```
d:\Quantum Projects\Paper_Implementation
├── Minimum-Error Quantum State Discrimination Based on Semidefinite Programming.pdf   # Original 2008 IEEE Conference Paper
├── README.md                                                                           # Detailed Documentation (This file)
├── requirements.txt                                                                    # Frozen dependency environment
├── test_discriminator.py                                                               # Comprehensive Pytest Unit Test Suite (7 tests)
├── main.ipynb                                                                          # Fully pre-executed demonstration & interactive notebook
│
├── quantum_state_discrimination/                                                       # Production Python Package
│   ├── __init__.py                                                                     # Public API exports
│   ├── core.py                                                                         # QuantumStateDiscriminator & DiscriminationResult
│   ├── states.py                                                                       # State generators, benchmarks, & noise channels
│   └── visualization.py                                                                # 3D Bloch sphere, optimality plots, & sweeps
│
└── scratch/                                                                            # Reproducibility scripts
    ├── build_notebook.py                                                               # Builds the complete main.ipynb notebook
    └── execute_notebook.py                                                             # Automated headless execution of the notebook
```

---

## 6. Installation & Environment Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### Step-by-Step Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/quantum-state-discrimination-sdp.git
   cd quantum-state-discrimination-sdp
   ```

2. **Create and activate a virtual environment**:
   - **Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify the installation**:
   ```bash
   python -c "import cvxpy, toqito, qutip, quantum_state_discrimination; print('Setup successful!')"
   ```

---

## 7. Quickstart Guide & Code Examples

### 7.1. Solving the Paper Benchmark

```python
from quantum_state_discrimination import QuantumStateDiscriminator, states, visualization

# 1. Load the benchmark states and prior probabilities
raw_states, probs = states.paper_2008_states()

# 2. Instantiate the discriminator
discriminator = QuantumStateDiscriminator(states=raw_states, probabilities=probs)

# 3. Solve using modern CVXPY SDP
result = discriminator.solve(method="cvxpy")

# 4. Print clean summary
visualization.display_results_summary(result)
```

**Console Output**:
```
=================================================================
  QUANTUM STATE DISCRIMINATION RESULTS (CVXPY (CLARABEL))
=================================================================
Hilbert Space Dimension (n):       2
Number of Quantum States (m):      3
Optimal Success Probability (Ps):   0.785410 (78.54%)
Minimum Error Probability (Pe):     0.214590 (21.46%)
Duality Gap:                       0.00e+00
Optimality Verification (YKL):     PASSED (Globally Optimal)
-----------------------------------------------------------------
Prior Probabilities xi:
  [0.1, 0.6, 0.3]
-----------------------------------------------------------------
Optimal POVM Measurement Operators {Pi_j}:

POVM Pi_1 (Trace = 0.0000):
[[ 0.  0.]
 [ 0.  0.]]

POVM Pi_2 (Trace = 1.0000):
[[0.7236 0.4472]
 [0.4472 0.2764]]

POVM Pi_3 (Trace = 1.0000):
[[ 0.2764 -0.4472]
 [-0.4472  0.7236]]
=================================================================
```

---

### 7.2. Multi-Solver Cross-Verification

Compare all three solvers on the same problem:

```python
for method in ["cvxpy", "toqito", "standard_sdp"]:
    res = discriminator.solve(method=method)
    print(f"[{res.solver_name}] -> Ps = {res.success_probability:.8f}, Duality Gap = {res.duality_gap:.2e}, Optimal = {res.is_optimal}")
```

---

### 7.3. Binary Discrimination vs. Analytical Helstrom Bound

```python
import numpy as np
from quantum_state_discrimination import QuantumStateDiscriminator

theta = np.pi / 6.0  # 30 degrees
psi1 = np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128)
psi2 = np.array([np.cos(theta), -np.sin(theta)], dtype=np.complex128)
priors = [0.35, 0.65]

disc = QuantumStateDiscriminator(states=[psi1, psi2], probabilities=priors)
res = disc.solve()

print(f"SDP Numerical Success Prob: {res.success_probability:.8f}")
print(f"Analytical Helstrom Bound:   {res.helstrom_bound:.8f}")
print(f"Difference:                  {abs(res.success_probability - res.helstrom_bound):.2e}")
```

---

### 7.4. Higher Dimensions: Qutrits ($n=3$) and Qudits ($n=4$)

```python
from quantum_state_discrimination import QuantumStateDiscriminator, states

# Discriminate 4 Haar-random pure qutrit states (dimension n=3)
qutrits, qutrit_probs = states.random_pure_states(num_states=4, dimension=3, seed=42)
disc_qutrit = QuantumStateDiscriminator(states=qutrits, probabilities=qutrit_probs)
res_qutrit = disc_qutrit.solve()
print(f"Qutrit Ensemble (m=4, n=3): Ps = {res_qutrit.success_probability:.6f}, Optimal = {res_qutrit.is_optimal}")

# Discriminate 5 random mixed qudit density matrices (dimension n=4)
qudits, qudit_probs = states.random_mixed_states(num_states=5, dimension=4, seed=42)
disc_qudit = QuantumStateDiscriminator(states=qudits, probabilities=qudit_probs)
res_qudit = disc_qudit.solve()
print(f"Qudit Mixed Ensemble (m=5, n=4): Ps = {res_qudit.success_probability:.6f}, Optimal = {res_qudit.is_optimal}")
```

---

### 7.5. Quantum Channel Decoherence (Depolarizing Noise)

```python
import numpy as np
import matplotlib.pyplot as plt
from quantum_state_discrimination import QuantumStateDiscriminator, states

trine, probs = states.trine_states()
noise_levels = np.linspace(0.0, 1.0, 20)
success_probs = []

for p in noise_levels:
    noisy_states = states.add_depolarizing_noise(trine, noise_parameter_p=p)
    disc = QuantumStateDiscriminator(states=noisy_states, probabilities=probs)
    res = disc.solve()
    success_probs.append(res.success_probability)

plt.plot(noise_levels, success_probs, 'b-o')
plt.xlabel("Depolarizing Noise Parameter p")
plt.ylabel("Success Probability Ps")
plt.title("Discrimination Degradation under Decoherence")
plt.grid(True)
plt.show()
```

---

### 7.6. QuTiP `Qobj` Interoperability

The discriminator accepts QuTiP quantum objects directly:

```python
import qutip as qt
from quantum_state_discrimination import QuantumStateDiscriminator

# Prepare QuTiP states
psi0 = qt.basis(2, 0)
psi1 = (qt.basis(2, 0) + qt.basis(2, 1)).unit()

disc = QuantumStateDiscriminator(states=[psi0, psi1], probabilities=[0.5, 0.5])
res = disc.solve()
print(f"QuTiP Input -> Ps: {res.success_probability:.6f}, Helstrom Match: {res.is_optimal}")
```

---

## 8. Running the Unit Test Suite

The test suite in [`test_discriminator.py`](test_discriminator.py) performs 7 rigorous unit tests:

1. `test_paper_2008_benchmark`: Exact numerical match with Wang et al. 2008 across CVXPY, Toqito, and Standard SDP.
2. `test_helstrom_bound_binary_states`: Verifies binary discrimination against the analytical Helstrom trace-norm formula.
3. `test_trine_states`: Confirms $P_s = 2/3$ for symmetric trine states.
4. `test_bb84_states`: Confirms $P_s = 1/2$ for the 4 BB84 states.
5. `test_qutrit_and_qudit_dynamic_states`: Tests dynamic random pure/mixed states in dimensions $n=3$ and $n=4$.
6. `test_qutip_qobj_input`: Verifies seamless compatibility with QuTiP `qt.Qobj` inputs.
7. `test_ykl_optimality_conditions`: Checks dual slack positive semidefiniteness, complementary slackness, and duality gap.

To run the tests:

```bash
pytest -v test_discriminator.py
```

**Test Execution Output**:
```
============================= test session starts =============================
platform win32 -- Python 3.13.2, pytest-9.1.1, pluggy-1.6.0
collected 7 items

test_discriminator.py::test_paper_2008_benchmark PASSED                  [ 14%]
test_discriminator.py::test_helstrom_bound_binary_states PASSED          [ 28%]
test_discriminator.py::test_trine_states PASSED                          [ 42%]
test_discriminator.py::test_bb84_states PASSED                           [ 57%]
test_discriminator.py::test_qutrit_and_qudit_dynamic_states PASSED       [ 71%]
test_discriminator.py::test_qutip_qobj_input PASSED                      [ 85%]
test_discriminator.py::test_ykl_optimality_conditions PASSED             [100%]

======================== 7 passed in 26.55s ========================
```

---

## 9. Interactive Jupyter Notebook (`main.ipynb`)

[`main.ipynb`](main.ipynb) is a pre-executed notebook containing:
- Mathematical derivations and Markdown formulations with LaTeX equations.
- Step-by-step benchmark reproduction of Section 5 of Wang et al. (2008).
- Multi-solver benchmarking tables.
- Visualizations:
  - 3D Bloch sphere projections showing quantum states and POVM vectors.
  - Dual slack eigenvalue spectrum and complementary slackness bar charts.
  - Prior probability sweeps and depolarizing decoherence curves.
- **Interactive IPyWidgets Dashboard**: Live GUI sliders allowing users to dynamically adjust prior probabilities $\xi_1, \xi_2, \xi_3$ and state orientation $\theta$, updating the optimal POVM elements and success probability in real time.

To launch the notebook:
```bash
jupyter notebook main.ipynb
```

---

## 10. References

1. **Wang, W., Dong, G., & Li, M.** (2008). *Minimum-Error Quantum State Discrimination Based on Semidefinite Programming*. In **Proceedings of the 27th Chinese Control Conference (CCC 2008)**, Kunming, China, pp. 521–524. IEEE.
2. **Helstrom, C. W.** (1976). *Quantum Detection and Estimation Theory*. Mathematics in Science and Engineering, Vol. 123, Academic Press, New York.
3. **Yuen, H. P., Kennedy, R. S., & Lax, M.** (1975). *Optimum testing of multiple hypotheses in quantum detection theory*. **IEEE Transactions on Information Theory**, 21(2), 125–134.
4. **Holevo, A. S.** (1973). *Statistical decision theory for quantum systems*. **Journal of Multivariate Analysis**, 3(4), 337–394.
5. **Eldar, Y. C., Megretski, A., & Verghese, G. C.** (2003). *Designing optimal quantum detectors via semidefinite programming*. **IEEE Transactions on Information Theory**, 49(4), 1007–1012.
6. **Barnett, S. M.** (2001). *Minimum-error discrimination between multiply symmetric states*. **Physical Review A**, 64(3), 030303(R).
7. **Andersson, E., Barnett, S. M., Gilson, C. R., & Hunter, K.** (2002). *Minimum-error discrimination between three mirror-symmetric states*. **Physical Review A**, 65(5), 052308.
8. **Boyd, S., & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge University Press.
9. **Nielsen, M. A., & Chuang, I. L.** (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

