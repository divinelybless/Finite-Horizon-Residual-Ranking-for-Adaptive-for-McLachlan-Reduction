"""
Constraint-Preserving Transient-Aware Adaptive Variational Simulation
of Non-Normal Transport--Diffusion Dynamics.

This is a noiseless state-vector prototype. It compares:
  M1: fixed reduced ansatz;
  M2: Euclidean adaptive constraint-preserving method;
  M3: transient-aware adaptive constraint-preserving method;
  M4: full-space penalty-boundary baseline.

All numerical outputs are saved in ``vqa_full_numerical_suite_results/``.
"""

from pathlib import Path
import itertools
import json
import math

import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.integrate import quad, quad_vec
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt



# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:  # Jupyter notebook execution
    SCRIPT_DIR = Path.cwd()

OUTPUT_DIR = SCRIPT_DIR / "vqa_full_numerical_suite_results"
FIG_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Main benchmark: n_h = 16 = 2^4 interior degrees of freedom.
N_INTERIOR = 16
A0 = 0.30
BETA = 2.00
D0 = 0.004
GAMMA0 = 0.05

DT = 0.002
T_FINAL = 4.0

# The ordinary Euclidean adaptive baseline and the transient-aware method are
# resource matched: same initial pool, candidate pool, time step, and maximum
# number of active tangent generators. Time evolution uses the unregularized
# minimum-norm pseudoinverse McLachlan update S^+ b. This matches the ideal
# residual identity used in the present numerical study.
BASE_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
MAX_ACTIVE_GENERATORS = 15
EUCLIDEAN_THRESHOLD = 0.02
TRANSIENT_THRESHOLD = 0.02
TAU_MAIN = 0.10

# Soft full-space boundary baseline.
LAMBDA_BC = 5.0

# Sensitivity studies.
HORIZON_GRID = [0.02, 0.05, 0.10, 0.20, 0.50]
BETA_GRID = [0.00, 0.50, 1.00, 1.50, 2.00]
DT_GRID = [0.004, 0.002, 0.001]
BUDGET_GRID = [4, 6, 8, 10, 12, 15]
DIMENSION_GRID = [8, 16, 32]
CONSTRAINED_BUDGET = 8
DIMENSION_BUDGET_FRACTION = 0.625

# Set these to False to omit expensive studies while debugging.
RUN_HORIZON_SWEEP = False
RUN_BUDGET_SWEEP = False
RUN_NONNORMALITY_SWEEP = False
RUN_TIME_STEP_SWEEP = False
RUN_DIMENSION_STUDY = False
GENERATE_PROFILE_FIGURE = False

np.set_printoptions(precision=8, suppress=True)




# -----------------------------------------------------------------------------
# 2. Finite-element assembly and mass whitening
# -----------------------------------------------------------------------------
def assemble_full_fem(n_interior, a0, beta, diffusion, gamma):
    """Assemble full P1 finite-element mass and transport--diffusion matrices."""
    n_full = n_interior + 2
    grid = np.linspace(0.0, 1.0, n_full)
    h = grid[1] - grid[0]

    mass = np.zeros((n_full, n_full), dtype=float)
    stiffness = np.zeros((n_full, n_full), dtype=float)
    quad_points, quad_weights = np.polynomial.legendre.leggauss(5)

    for element in range(n_full - 1):
        x_left = grid[element]
        x_right = grid[element + 1]
        element_nodes = (element, element + 1)

        for local_row, global_row in enumerate(element_nodes):
            for local_col, global_col in enumerate(element_nodes):
                local_mass = 0.0
                local_stiffness = 0.0

                for xi, weight in zip(quad_points, quad_weights):
                    x_quad = 0.5 * (x_left + x_right) + 0.5 * (x_right - x_left) * xi
                    jacobian = 0.5 * (x_right - x_left)
                    phi = ((x_right - x_quad) / h, (x_quad - x_left) / h)
                    dphi = (-1.0 / h, 1.0 / h)
                    velocity = a0 - beta * (x_quad - 0.5)

                    local_mass += weight * jacobian * phi[local_col] * phi[local_row]
                    local_stiffness += weight * jacobian * (
                        diffusion * dphi[local_col] * dphi[local_row]
                        - velocity * phi[local_col] * dphi[local_row]
                        + gamma * phi[local_col] * phi[local_row]
                    )

                mass[global_row, global_col] += local_mass
                stiffness[global_row, global_col] += local_stiffness

    return grid, mass, stiffness


def spd_square_root_and_inverse(matrix):
    """Return M^(1/2) and M^(-1/2) for a symmetric positive-definite matrix."""
    eigenvalues, eigenvectors = la.eigh(matrix)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError("Mass matrix is not positive definite.")

    square_root = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_square_root = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    return square_root, inverse_square_root


def build_problem(n_interior, a0, beta, diffusion, gamma, lambda_bc):
    """Build the reduced Dirichlet system and the full-space penalty baseline."""
    grid, mass_full, stiffness_full = assemble_full_fem(
        n_interior, a0, beta, diffusion, gamma
    )

    mass_int = mass_full[1:-1, 1:-1]
    stiffness_int = stiffness_full[1:-1, 1:-1]
    mass_half, mass_inv_half = spd_square_root_and_inverse(mass_int)
    generator = -mass_inv_half @ stiffness_int @ mass_inv_half

    mass_full_half, mass_full_inv_half = spd_square_root_and_inverse(mass_full)

    boundary_operator = np.zeros((2, n_interior + 2), dtype=float)
    boundary_operator[0, 0] = 1.0
    boundary_operator[1, -1] = 1.0

    penalty_generator = -mass_full_inv_half @ (
        stiffness_full + lambda_bc * boundary_operator.T @ boundary_operator
    ) @ mass_full_inv_half

    initial_interior = (
        np.sin(np.pi * grid[1:-1])
        + 0.35 * np.sin(4.0 * np.pi * grid[1:-1])
    )
    initial_whitened = mass_half @ initial_interior

    initial_full = np.zeros(n_interior + 2, dtype=float)
    initial_full[1:-1] = initial_interior
    initial_full_whitened = mass_full_half @ initial_full

    return {
        "x": grid,
        "M_full": mass_full,
        "K_full": stiffness_full,
        "M_int": mass_int,
        "K_int": stiffness_int,
        "M_half": mass_half,
        "M_inv_half": mass_inv_half,
        "L": generator,
        "B": boundary_operator,
        "M_full_half": mass_full_half,
        "M_full_inv_half": mass_full_inv_half,
        "L_penalty": penalty_generator,
        "y0": initial_whitened,
        "y0_full": initial_full_whitened,
    }




# -----------------------------------------------------------------------------
# 3. Norms, transient-energy metric, and tangent-space utilities
# -----------------------------------------------------------------------------
_WEIGHT_FACTOR_CACHE = {}


def euclidean_norm(vector):
    return float(np.linalg.norm(vector))


def weight_factor(weight):
    """Return a cached factor R with weight = R^T R."""
    key = id(weight)
    if key not in _WEIGHT_FACTOR_CACHE:
        try:
            factor = la.cholesky(weight, lower=False)
        except la.LinAlgError:
            eigenvalues, eigenvectors = la.eigh(weight)
            eigenvalues = np.maximum(eigenvalues, 1.0e-14)
            factor = np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        _WEIGHT_FACTOR_CACHE[key] = factor
    return _WEIGHT_FACTOR_CACHE[key]


def weighted_norm(vector, weight):
    return float(np.sqrt(max(float(vector.T @ weight @ vector), 0.0)))


def transient_energy_matrix(generator, horizon):
    r"""Compute W_tau = int_0^tau exp(s L^dagger) exp(s L) ds."""
    if horizon <= 0.0:
        raise ValueError("The prediction horizon tau must be strictly positive.")

    def integrand(time):
        return la.expm(time * generator.conj().T) @ la.expm(time * generator)

    metric, _ = quad_vec(integrand, 0.0, float(horizon), epsabs=1e-10, epsrel=1e-10)
    metric = np.real_if_close(0.5 * (metric + metric.T))

    # Remove only roundoff-level negative eigenvalues from numerical quadrature.
    eigenvalues, eigenvectors = la.eigh(metric)
    eigenvalues = np.maximum(eigenvalues, 1.0e-14)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def apply_givens_generator(vector, pair):
    """Apply a real two-level skew-symmetric Givens tangent generator."""
    i, j = pair
    result = np.zeros_like(vector)
    result[i] = -vector[j]
    result[j] = vector[i]
    return result


def tangent_matrix(state, active_pairs):
    if not active_pairs:
        return np.zeros((state.size, 0), dtype=float)
    return np.column_stack([apply_givens_generator(state, pair) for pair in active_pairs])


def project_real_tangent(vector, tangent, weight=None):
    """Project onto a real tangent span using Euclidean or W-weighted geometry.

    The least-squares problem is solved directly, rather than through the
    normal equations. This is important because forming the Gram matrix squares
    the condition number and can alter the numerical rank of nearly dependent
    tangent directions.
    """
    if tangent.shape[1] == 0:
        return np.zeros_like(vector)

    if weight is None:
        coefficients = la.lstsq(tangent, vector)[0]
    else:
        upper = weight_factor(weight)
        coefficients = la.lstsq(upper @ tangent, upper @ vector)[0]
    return tangent @ coefficients


def normalized_vector_field(state, generator):
    """Return phase-fixed normalized flow G(phi) and complex growth rate ell(phi)."""
    ell = float(state @ generator @ state)
    flow = generator @ state - ell * state
    return flow, ell


def tangent_diagnostics(state, generator, active_pairs, weight=None):
    """Return tangent projection, residual, and tangent matrix in a chosen metric."""
    flow, ell = normalized_vector_field(state, generator)
    tangent = tangent_matrix(state, active_pairs)
    projection = project_real_tangent(flow, tangent, weight=weight)
    residual = flow - projection
    return {
        "flow": flow,
        "ell": ell,
        "tangent": tangent,
        "projection": projection,
        "residual": residual,
    }


def candidate_scores(state, active_pairs, tangent, residual, candidates, weight=None):
    """Compute Euclidean or finite-horizon greedy residual-reduction scores."""
    scores = {}

    for pair in candidates:
        if pair in active_pairs:
            continue

        candidate_direction = apply_givens_generator(state, pair)
        represented_component = project_real_tangent(
            candidate_direction, tangent, weight=weight
        )
        unresolved_direction = candidate_direction - represented_component

        if weight is None:
            denominator = euclidean_norm(unresolved_direction) ** 2
            numerator = unresolved_direction @ residual
        else:
            denominator = weighted_norm(unresolved_direction, weight) ** 2
            numerator = unresolved_direction @ weight @ residual

        if denominator > 1.0e-13:
            scores[pair] = float((numerator ** 2) / denominator)

    return scores




# -----------------------------------------------------------------------------
# 4. Variational simulation
# -----------------------------------------------------------------------------
def simulate_variational(
    generator,
    initial_state,
    dt,
    final_time,
    base_pairs,
    mode,
    transient_metric=None,
    threshold=0.02,
    max_active_generators=15,
):
    """
    Simulate a local tangent-space variational method.

    mode = "fixed"      : no enrichment;
    mode = "euclidean"  : Euclidean residual trigger and score;
    mode = "transient"  : finite-horizon trigger and score, but Euclidean
                           McLachlan projection for time propagation.
    """
    if mode not in {"fixed", "euclidean", "transient"}:
        raise ValueError("mode must be 'fixed', 'euclidean', or 'transient'.")
    if mode == "transient" and transient_metric is None:
        raise ValueError("Transient-aware selection requires W_tau.")

    initial_norm = euclidean_norm(initial_state)
    if initial_norm == 0.0:
        raise ValueError("Initial state must be nonzero.")

    state = initial_state.astype(float) / initial_norm
    log_magnitude = float(np.log(initial_norm))
    phase = 0.0

    active_pairs = list(base_pairs)
    all_pairs = list(itertools.combinations(range(initial_state.size), 2))

    n_steps = int(round(final_time / dt))
    times = np.arange(n_steps + 1, dtype=float) * dt

    state_history = np.empty((n_steps + 1, initial_state.size), dtype=float)
    state_history[0] = initial_state

    euclidean_residual = np.empty(n_steps + 1, dtype=float)
    relative_euclidean_residual = np.empty(n_steps + 1, dtype=float)
    transient_residual = np.full(n_steps + 1, np.nan, dtype=float)
    relative_transient_residual = np.full(n_steps + 1, np.nan, dtype=float)
    active_counts = np.empty(n_steps + 1, dtype=int)
    selected_generators = []

    def euclidean_rhs(current_state):
        diagnostics = tangent_diagnostics(
            current_state, generator, active_pairs, weight=None
        )
        return diagnostics["projection"], diagnostics["ell"], diagnostics

    for step in range(n_steps):
        euclidean = tangent_diagnostics(state, generator, active_pairs, weight=None)
        euclidean_residual[step] = euclidean_norm(euclidean["residual"])
        relative_euclidean_residual[step] = euclidean_residual[step] / (
            euclidean_norm(euclidean["flow"]) + 1.0e-12
        )
        active_counts[step] = len(active_pairs)

        if transient_metric is not None:
            transient = tangent_diagnostics(
                state, generator, active_pairs, weight=transient_metric
            )
            transient_residual[step] = weighted_norm(
                transient["residual"], transient_metric
            )
            relative_transient_residual[step] = transient_residual[step] / (
                weighted_norm(transient["flow"], transient_metric) + 1.0e-12
            )
        else:
            transient = None

        # Adaptive enrichment is applied before the RK4 state update.
        should_enrich = False
        score_weight = None
        score_residual = None
        score_tangent = None

        if mode == "euclidean":
            should_enrich = relative_euclidean_residual[step] > threshold
            score_residual = euclidean["residual"]
            score_tangent = euclidean["tangent"]
        elif mode == "transient":
            should_enrich = relative_transient_residual[step] > threshold
            score_weight = transient_metric
            score_residual = transient["residual"]
            score_tangent = transient["tangent"]

        if should_enrich and len(active_pairs) < max_active_generators:
            scores = candidate_scores(
                state,
                active_pairs,
                score_tangent,
                score_residual,
                all_pairs,
                weight=score_weight,
            )
            if scores:
                selected_pair, selected_score = max(scores.items(), key=lambda item: item[1])
                if selected_score > 1.0e-13:
                    active_pairs.append(selected_pair)
                    selected_generators.append(
                        {
                            "time": float(times[step]),
                            "pair_i": int(selected_pair[0]),
                            "pair_j": int(selected_pair[1]),
                            "score": float(selected_score),
                        }
                    )

                    # Recompute diagnostics after changing the tangent space.
                    euclidean = tangent_diagnostics(state, generator, active_pairs, weight=None)
                    euclidean_residual[step] = euclidean_norm(euclidean["residual"])
                    relative_euclidean_residual[step] = euclidean_residual[step] / (
                        euclidean_norm(euclidean["flow"]) + 1.0e-12
                    )
                    active_counts[step] = len(active_pairs)

                    if transient_metric is not None:
                        transient = tangent_diagnostics(
                            state, generator, active_pairs, weight=transient_metric
                        )
                        transient_residual[step] = weighted_norm(
                            transient["residual"], transient_metric
                        )
                        relative_transient_residual[step] = transient_residual[step] / (
                            weighted_norm(transient["flow"], transient_metric) + 1.0e-12
                        )

        # Standard Euclidean McLachlan propagation with fourth-order RK4.
        k1, ell1, _ = euclidean_rhs(state)

        stage = state + 0.5 * dt * k1
        stage /= euclidean_norm(stage)
        k2, ell2, _ = euclidean_rhs(stage)

        stage = state + 0.5 * dt * k2
        stage /= euclidean_norm(stage)
        k3, ell3, _ = euclidean_rhs(stage)

        stage = state + dt * k3
        stage /= euclidean_norm(stage)
        k4, ell4, _ = euclidean_rhs(stage)

        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        state /= euclidean_norm(state)

        log_magnitude += (dt / 6.0) * np.real(ell1 + 2.0 * ell2 + 2.0 * ell3 + ell4)
        phase += (dt / 6.0) * np.imag(ell1 + 2.0 * ell2 + 2.0 * ell3 + ell4)

        state_history[step + 1] = np.exp(log_magnitude) * state

    # Final diagnostics.
    euclidean = tangent_diagnostics(state, generator, active_pairs, weight=None)
    euclidean_residual[-1] = euclidean_norm(euclidean["residual"])
    relative_euclidean_residual[-1] = euclidean_residual[-1] / (
        euclidean_norm(euclidean["flow"]) + 1.0e-12
    )
    active_counts[-1] = len(active_pairs)

    if transient_metric is not None:
        transient = tangent_diagnostics(state, generator, active_pairs, weight=transient_metric)
        transient_residual[-1] = weighted_norm(transient["residual"], transient_metric)
        relative_transient_residual[-1] = transient_residual[-1] / (
            weighted_norm(transient["flow"], transient_metric) + 1.0e-12
        )

    return {
        "t": times,
        "y": state_history,
        "euclidean_residual": euclidean_residual,
        "relative_euclidean_residual": relative_euclidean_residual,
        "transient_residual": transient_residual,
        "relative_transient_residual": relative_transient_residual,
        "active_count": active_counts,
        "active_pairs": active_pairs,
        "selected_generators": selected_generators,
    }




# -----------------------------------------------------------------------------
# 5. Reconstruction, errors, and residual certificate
# -----------------------------------------------------------------------------
def embed_in_full_vector(interior_vectors):
    full_vectors = np.zeros((interior_vectors.shape[0], N_INTERIOR + 2), dtype=float)
    full_vectors[:, 1:-1] = interior_vectors
    return full_vectors


def mass_relative_error(reference_full, approximation_full, mass_full):
    difference = reference_full - approximation_full
    numerator = np.sqrt(
        np.maximum(
            np.einsum("bi,ij,bj->b", difference, mass_full, difference),
            0.0,
        )
    )
    denominator = np.sqrt(
        np.maximum(
            np.einsum("bi,ij,bj->b", reference_full, mass_full, reference_full),
            1.0e-30,
        )
    )
    return numerator / denominator


def piecewise_linear_residual_certificate(state_history, generator, dt, numerical_abscissa):
    """
    Compute the continuous residual bound for the piecewise-linear interpolation
    of a time-discrete state trajectory. This includes time-stepping residual.
    """
    certificate = np.zeros(state_history.shape[0], dtype=float)
    amplification = math.exp(numerical_abscissa * dt)

    for step in range(state_history.shape[0] - 1):
        y_left = state_history[step]
        delta_y = state_history[step + 1] - y_left
        derivative = delta_y / dt

        def integrand(local_time):
            interpolant = y_left + (local_time / dt) * delta_y
            residual = derivative - generator @ interpolant
            return math.exp(numerical_abscissa * (dt - local_time)) * euclidean_norm(residual)

        local_contribution = quad(
            integrand, 0.0, dt, epsabs=1.0e-10, epsrel=1.0e-10, limit=100
        )[0]
        certificate[step + 1] = amplification * certificate[step] + local_contribution

    return certificate


def selected_generators_frame(result):
    if result["selected_generators"]:
        return pd.DataFrame(result["selected_generators"])
    return pd.DataFrame(columns=["time", "pair_i", "pair_j", "score"])


def save_figure(file_stem):
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{file_stem}.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{file_stem}.png", dpi=300, bbox_inches="tight")
    plt.close()




# -----------------------------------------------------------------------------
# 6. Main benchmark
# -----------------------------------------------------------------------------
def run_main_benchmark():
    problem = build_problem(
        N_INTERIOR, A0, BETA, D0, GAMMA0, LAMBDA_BC
    )

    generator = problem["L"]
    prediction_metric = transient_energy_matrix(generator, TAU_MAIN)

    fixed = simulate_variational(
        generator,
        problem["y0"],
        DT,
        T_FINAL,
        BASE_PAIRS,
        mode="fixed",
        transient_metric=prediction_metric,
        threshold=np.inf,
        max_active_generators=len(BASE_PAIRS),
    )

    euclidean_adaptive = simulate_variational(
        generator,
        problem["y0"],
        DT,
        T_FINAL,
        BASE_PAIRS,
        mode="euclidean",
        transient_metric=prediction_metric,
        threshold=EUCLIDEAN_THRESHOLD,
        max_active_generators=MAX_ACTIVE_GENERATORS,
    )

    transient_adaptive = simulate_variational(
        generator,
        problem["y0"],
        DT,
        T_FINAL,
        BASE_PAIRS,
        mode="transient",
        transient_metric=prediction_metric,
        threshold=TRANSIENT_THRESHOLD,
        max_active_generators=MAX_ACTIVE_GENERATORS,
    )

    penalty = simulate_variational(
        problem["L_penalty"],
        problem["y0_full"],
        DT,
        T_FINAL,
        BASE_PAIRS,
        mode="euclidean",
        transient_metric=None,
        threshold=EUCLIDEAN_THRESHOLD,
        max_active_generators=MAX_ACTIVE_GENERATORS + 2,
    )

    times = euclidean_adaptive["t"]
    reference = expm_multiply(
        generator,
        problem["y0"],
        start=0.0,
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
    )

    z_reference_int = (problem["M_inv_half"] @ reference.T).T
    z_fixed_int = (problem["M_inv_half"] @ fixed["y"].T).T
    z_euclidean_int = (problem["M_inv_half"] @ euclidean_adaptive["y"].T).T
    z_transient_int = (problem["M_inv_half"] @ transient_adaptive["y"].T).T

    z_reference_full = embed_in_full_vector(z_reference_int)
    z_fixed_full = embed_in_full_vector(z_fixed_int)
    z_euclidean_full = embed_in_full_vector(z_euclidean_int)
    z_transient_full = embed_in_full_vector(z_transient_int)
    z_penalty_full = (problem["M_full_inv_half"] @ penalty["y"].T).T

    reference_norm = np.linalg.norm(reference, axis=1)
    relative_error_fixed = np.linalg.norm(reference - fixed["y"], axis=1) / reference_norm
    relative_error_euclidean = (
        np.linalg.norm(reference - euclidean_adaptive["y"], axis=1) / reference_norm
    )
    relative_error_transient = (
        np.linalg.norm(reference - transient_adaptive["y"], axis=1) / reference_norm
    )
    relative_error_penalty = mass_relative_error(
        z_reference_full, z_penalty_full, problem["M_full"]
    )

    boundary_error_fixed = np.linalg.norm((problem["B"] @ z_fixed_full.T).T, axis=1)
    boundary_error_euclidean = np.linalg.norm(
        (problem["B"] @ z_euclidean_full.T).T, axis=1
    )
    boundary_error_transient = np.linalg.norm(
        (problem["B"] @ z_transient_full.T).T, axis=1
    )
    boundary_error_penalty = np.linalg.norm(
        (problem["B"] @ z_penalty_full.T).T, axis=1
    )

    spectral_abscissa = float(np.max(la.eigvals(generator).real))
    numerical_abscissa = float(np.max(la.eigvalsh((generator + generator.T) / 2.0)))
    nonnormality = float(la.norm(generator @ generator.T - generator.T @ generator, ord=2))

    certificate_euclidean = piecewise_linear_residual_certificate(
        euclidean_adaptive["y"], generator, DT, numerical_abscissa
    )
    certificate_transient = piecewise_linear_residual_certificate(
        transient_adaptive["y"], generator, DT, numerical_abscissa
    )

    absolute_error_euclidean = np.linalg.norm(reference - euclidean_adaptive["y"], axis=1)
    absolute_error_transient = np.linalg.norm(reference - transient_adaptive["y"], axis=1)

    summary = pd.DataFrame(
        {
            "method": [
                "Fixed reduced ansatz",
                "Euclidean adaptive constraint-preserving",
                "Transient-aware adaptive constraint-preserving",
                "Penalty-boundary baseline",
            ],
            "max_relative_error": [
                float(relative_error_fixed.max()),
                float(relative_error_euclidean.max()),
                float(relative_error_transient.max()),
                float(relative_error_penalty.max()),
            ],
            "final_relative_error": [
                float(relative_error_fixed[-1]),
                float(relative_error_euclidean[-1]),
                float(relative_error_transient[-1]),
                float(relative_error_penalty[-1]),
            ],
            "max_boundary_violation": [
                float(boundary_error_fixed.max()),
                float(boundary_error_euclidean.max()),
                float(boundary_error_transient.max()),
                float(boundary_error_penalty.max()),
            ],
            "final_active_generators": [
                int(fixed["active_count"][-1]),
                int(euclidean_adaptive["active_count"][-1]),
                int(transient_adaptive["active_count"][-1]),
                int(penalty["active_count"][-1]),
            ],
            "number_of_added_generators": [
                len(fixed["selected_generators"]),
                len(euclidean_adaptive["selected_generators"]),
                len(transient_adaptive["selected_generators"]),
                len(penalty["selected_generators"]),
            ],
        }
    )

    # Save main numerical data.
    summary.to_csv(DATA_DIR / "main_method_summary.csv", index=False)
    selected_generators_frame(euclidean_adaptive).to_csv(
        DATA_DIR / "euclidean_selected_generators.csv", index=False
    )
    selected_generators_frame(transient_adaptive).to_csv(
        DATA_DIR / "transient_selected_generators.csv", index=False
    )

    main_time_series = pd.DataFrame(
        {
            "time": times,
            "relative_error_fixed": relative_error_fixed,
            "relative_error_euclidean": relative_error_euclidean,
            "relative_error_transient": relative_error_transient,
            "relative_error_penalty": relative_error_penalty,
            "boundary_error_fixed": boundary_error_fixed,
            "boundary_error_euclidean": boundary_error_euclidean,
            "boundary_error_transient": boundary_error_transient,
            "boundary_error_penalty": boundary_error_penalty,
            "active_count_euclidean": euclidean_adaptive["active_count"],
            "active_count_transient": transient_adaptive["active_count"],
        }
    )
    main_time_series.to_csv(DATA_DIR / "main_time_series.csv", index=False)

    np.savez_compressed(
        DATA_DIR / "main_benchmark_arrays.npz",
        t=times,
        x=problem["x"],
        y_reference=reference,
        y_fixed=fixed["y"],
        y_euclidean=euclidean_adaptive["y"],
        y_transient=transient_adaptive["y"],
        y_penalty=penalty["y"],
        z_reference_full=z_reference_full,
        z_fixed_full=z_fixed_full,
        z_euclidean_full=z_euclidean_full,
        z_transient_full=z_transient_full,
        z_penalty_full=z_penalty_full,
        relative_error_fixed=relative_error_fixed,
        relative_error_euclidean=relative_error_euclidean,
        relative_error_transient=relative_error_transient,
        relative_error_penalty=relative_error_penalty,
        boundary_error_fixed=boundary_error_fixed,
        boundary_error_euclidean=boundary_error_euclidean,
        boundary_error_transient=boundary_error_transient,
        boundary_error_penalty=boundary_error_penalty,
        euclidean_tangent_residual=euclidean_adaptive["euclidean_residual"],
        transient_tangent_residual=transient_adaptive["transient_residual"],
        certificate_euclidean=certificate_euclidean,
        certificate_transient=certificate_transient,
        absolute_error_euclidean=absolute_error_euclidean,
        absolute_error_transient=absolute_error_transient,
        active_count_euclidean=euclidean_adaptive["active_count"],
        active_count_transient=transient_adaptive["active_count"],
    )

    certificate_verification = pd.DataFrame(
        {
            "method": ["Euclidean adaptive", "Transient-aware adaptive"],
            "certificate_dominates_all_sampled_nodes": [
                bool(np.all(certificate_euclidean >= absolute_error_euclidean - 1.0e-10)),
                bool(np.all(certificate_transient >= absolute_error_transient - 1.0e-10)),
            ],
            "minimum_certificate_gap": [
                float(np.min(certificate_euclidean - absolute_error_euclidean)),
                float(np.min(certificate_transient - absolute_error_transient)),
            ],
            "final_sampled_error": [
                float(absolute_error_euclidean[-1]),
                float(absolute_error_transient[-1]),
            ],
            "final_certificate": [
                float(certificate_euclidean[-1]),
                float(certificate_transient[-1]),
            ],
        }
    )
    certificate_verification.to_csv(DATA_DIR / "certificate_verification.csv", index=False)

    # ---------- Main figures ----------
    reference_growth = np.linalg.norm(reference, axis=1) / euclidean_norm(problem["y0"])
    euclidean_growth = np.linalg.norm(euclidean_adaptive["y"], axis=1) / euclidean_norm(problem["y0"])
    transient_growth = np.linalg.norm(transient_adaptive["y"], axis=1) / euclidean_norm(problem["y0"])

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(times, reference_growth, label="Reference")
    plt.plot(times, euclidean_growth, label="Euclidean adaptive")
    plt.plot(times, transient_growth, label="Transient-aware adaptive")
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"$\|\mathbf{y}(t)\|_2/\|\mathbf{y}_0\|_2$")
    plt.title("Transient amplification in the non-normal benchmark")
    plt.legend()
    save_figure("transient_growth_comparison")

    # The profile figure is disabled by default pending separate validation of
    # the full-space penalty reconstruction against the reduced reference.
    if GENERATE_PROFILE_FIGURE:
        peak_index = int(np.argmax(reference_growth))
        peak_time = times[peak_index]
        plt.figure(figsize=(7.0, 4.5))
        plt.plot(problem["x"], np.real(z_reference_full[peak_index]), label="Reference")
        plt.plot(problem["x"], np.real(z_fixed_full[peak_index]), label="Fixed reduced")
        plt.plot(problem["x"], np.real(z_euclidean_full[peak_index]), label="Euclidean adaptive")
        plt.plot(problem["x"], np.real(z_transient_full[peak_index]), label="Transient-aware adaptive")
        plt.plot(problem["x"], np.real(z_penalty_full[peak_index]), label="Penalty baseline")
        plt.xlabel(r"Spatial coordinate $x$")
        plt.ylabel(r"$u_h(x,t)$")
        plt.title(f"Solution profile at peak transient growth, t = {peak_time:.3f}")
        plt.legend()
        save_figure("solution_profile_peak")

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(times, relative_error_fixed, label="Fixed reduced")
    plt.plot(times, relative_error_euclidean, label="Euclidean adaptive")
    plt.plot(times, relative_error_transient, label="Transient-aware adaptive")
    plt.plot(times, relative_error_penalty, label="Penalty baseline")
    plt.yscale("log")
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"Relative physical $L^2$ error")
    plt.title("Main comparison of variational approximation errors")
    plt.legend()
    save_figure("relative_errors_main")

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(times, np.maximum(boundary_error_fixed, 1.0e-16), label="Fixed reduced")
    plt.plot(times, np.maximum(boundary_error_euclidean, 1.0e-16), label="Euclidean adaptive")
    plt.plot(times, np.maximum(boundary_error_transient, 1.0e-16), label="Transient-aware adaptive")
    plt.plot(times, np.maximum(boundary_error_penalty, 1.0e-16), label="Penalty baseline")
    plt.yscale("log")
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"$\|B_h\widehat{\overline{\mathbf{z}}}(t)\|_2$")
    plt.title("Boundary-condition violation")
    plt.legend()
    save_figure("boundary_violation_main")

    plt.figure(figsize=(7.0, 4.5))
    plt.step(times, euclidean_adaptive["active_count"], where="post", label="Euclidean adaptive")
    plt.step(times, transient_adaptive["active_count"], where="post", label="Transient-aware adaptive")
    plt.xlabel(r"Time $t$")
    plt.ylabel("Number of active generators")
    plt.title("Resource-matched adaptive ansatz growth")
    plt.legend()
    save_figure("adaptive_complexity_comparison")

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(times, euclidean_adaptive["euclidean_residual"], label="Euclidean residual: M2")
    plt.plot(times, transient_adaptive["transient_residual"], label="Finite-horizon residual: M3")
    plt.yscale("log")
    plt.xlabel(r"Time $t$")
    plt.ylabel("Residual magnitude")
    plt.title("Residual diagnostics for the two adaptive strategies")
    plt.legend()
    save_figure("adaptive_residual_diagnostics")

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(times, absolute_error_euclidean, label="True error: Euclidean adaptive")
    plt.plot(times, certificate_euclidean, label="Certificate: Euclidean adaptive")
    plt.plot(times, absolute_error_transient, label="True error: transient-aware adaptive")
    plt.plot(times, certificate_transient, label="Certificate: transient-aware adaptive")
    plt.yscale("log")
    plt.xlabel(r"Time $t$")
    plt.ylabel("Absolute mass-whitened error")
    plt.title("Piecewise-linear residual certificate")
    plt.legend()
    save_figure("residual_certificate_comparison")

    return {
        "problem": problem,
        "times": times,
        "reference": reference,
        "fixed": fixed,
        "euclidean": euclidean_adaptive,
        "transient": transient_adaptive,
        "penalty": penalty,
        "summary": summary,
        "relative_error_fixed": relative_error_fixed,
        "relative_error_euclidean": relative_error_euclidean,
        "relative_error_transient": relative_error_transient,
        "relative_error_penalty": relative_error_penalty,
        "spectral_abscissa": spectral_abscissa,
        "numerical_abscissa": numerical_abscissa,
        "nonnormality": nonnormality,
        "certificate_euclidean": certificate_euclidean,
        "certificate_transient": certificate_transient,
        "absolute_error_euclidean": absolute_error_euclidean,
        "absolute_error_transient": absolute_error_transient,
    }




# -----------------------------------------------------------------------------
# 7. Sensitivity experiments
# -----------------------------------------------------------------------------
def run_horizon_sensitivity(problem):
    rows = []
    generator = problem["L"]
    reference_for_euclidean = expm_multiply(
        generator,
        problem["y0"],
        start=0.0,
        stop=T_FINAL,
        num=int(round(T_FINAL / DT)) + 1,
        endpoint=True,
    )
    euclidean_reference = simulate_variational(
        generator,
        problem["y0"],
        DT,
        T_FINAL,
        BASE_PAIRS,
        mode="euclidean",
        transient_metric=None,
        threshold=EUCLIDEAN_THRESHOLD,
        max_active_generators=MAX_ACTIVE_GENERATORS,
    )
    euclidean_reference_error = np.linalg.norm(reference_for_euclidean - euclidean_reference["y"], axis=1) / np.linalg.norm(reference_for_euclidean, axis=1)

    for horizon in HORIZON_GRID:
        metric = transient_energy_matrix(generator, horizon)
        result = simulate_variational(
            generator,
            problem["y0"],
            DT,
            T_FINAL,
            BASE_PAIRS,
            mode="transient",
            transient_metric=metric,
            threshold=TRANSIENT_THRESHOLD,
            max_active_generators=MAX_ACTIVE_GENERATORS,
        )
        reference = expm_multiply(
            generator,
            problem["y0"],
            start=0.0,
            stop=float(result["t"][-1]),
            num=len(result["t"]),
            endpoint=True,
        )
        relative_error = np.linalg.norm(reference - result["y"], axis=1) / np.linalg.norm(reference, axis=1)

        rows.append(
            {
                "tau": horizon,
                "max_relative_error": float(relative_error.max()),
                "final_relative_error": float(relative_error[-1]),
                "final_active_generators": int(result["active_count"][-1]),
                "number_of_added_generators": len(result["selected_generators"]),
                "time_of_last_enrichment": (
                    float(result["selected_generators"][-1]["time"])
                    if result["selected_generators"] else np.nan
                ),
                "euclidean_max_relative_error": float(euclidean_reference_error.max()),
                "euclidean_final_relative_error": float(euclidean_reference_error[-1]),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "horizon_sensitivity.csv", index=False)

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(frame["tau"], frame["max_relative_error"], label="Transient-aware adaptive")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Prediction horizon $\tau$")
    plt.ylabel(r"Maximum relative $L^2$ error")
    plt.title("Finite-horizon sensitivity")
    plt.legend()
    save_figure("horizon_sensitivity")

    return frame


def run_nonnormality_sweep():
    rows = []

    for beta in BETA_GRID:
        problem = build_problem(
            N_INTERIOR, A0, beta, D0, GAMMA0, LAMBDA_BC
        )
        generator = problem["L"]
        metric = transient_energy_matrix(generator, TAU_MAIN)

        euclidean = simulate_variational(
            generator,
            problem["y0"],
            DT,
            T_FINAL,
            BASE_PAIRS,
            mode="euclidean",
            transient_metric=metric,
            threshold=EUCLIDEAN_THRESHOLD,
            max_active_generators=CONSTRAINED_BUDGET,
        )
        transient = simulate_variational(
            generator,
            problem["y0"],
            DT,
            T_FINAL,
            BASE_PAIRS,
            mode="transient",
            transient_metric=metric,
            threshold=TRANSIENT_THRESHOLD,
            max_active_generators=CONSTRAINED_BUDGET,
        )

        reference = expm_multiply(
            generator,
            problem["y0"],
            start=0.0,
            stop=float(euclidean["t"][-1]),
            num=len(euclidean["t"]),
            endpoint=True,
        )
        euclidean_error = np.linalg.norm(reference - euclidean["y"], axis=1) / np.linalg.norm(reference, axis=1)
        transient_error = np.linalg.norm(reference - transient["y"], axis=1) / np.linalg.norm(reference, axis=1)

        rows.append(
            {
                "beta": beta,
                "spectral_abscissa": float(np.max(la.eigvals(generator).real)),
                "numerical_abscissa": float(np.max(la.eigvalsh((generator + generator.T) / 2.0))),
                "nonnormality_measure": float(la.norm(generator @ generator.T - generator.T @ generator, ord=2)),
                "max_relative_error_euclidean": float(euclidean_error.max()),
                "max_relative_error_transient": float(transient_error.max()),
                "active_generators_euclidean": int(euclidean["active_count"][-1]),
                "active_generators_transient": int(transient["active_count"][-1]),
                "max_active_generators": CONSTRAINED_BUDGET,
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "nonnormality_sweep.csv", index=False)

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(frame["beta"], frame["max_relative_error_euclidean"], label="Euclidean adaptive")
    plt.plot(frame["beta"], frame["max_relative_error_transient"], label="Transient-aware adaptive")
    plt.yscale("log")
    plt.xlabel(r"Compression parameter $\beta$")
    plt.ylabel(r"Maximum relative $L^2$ error")
    plt.title("Error across increasing non-normal transport strength")
    plt.legend()
    save_figure("nonnormality_sweep")

    return frame


def run_time_step_sensitivity(problem):
    rows = []
    generator = problem["L"]
    metric = transient_energy_matrix(generator, TAU_MAIN)

    for dt in DT_GRID:
        euclidean = simulate_variational(
            generator,
            problem["y0"],
            dt,
            T_FINAL,
            BASE_PAIRS,
            mode="euclidean",
            transient_metric=metric,
            threshold=EUCLIDEAN_THRESHOLD,
            max_active_generators=MAX_ACTIVE_GENERATORS,
        )
        transient = simulate_variational(
            generator,
            problem["y0"],
            dt,
            T_FINAL,
            BASE_PAIRS,
            mode="transient",
            transient_metric=metric,
            threshold=TRANSIENT_THRESHOLD,
            max_active_generators=MAX_ACTIVE_GENERATORS,
        )

        reference = expm_multiply(
            generator,
            problem["y0"],
            start=0.0,
            stop=float(euclidean["t"][-1]),
            num=len(euclidean["t"]),
            endpoint=True,
        )
        euclidean_error = np.linalg.norm(reference - euclidean["y"], axis=1) / np.linalg.norm(reference, axis=1)
        transient_error = np.linalg.norm(reference - transient["y"], axis=1) / np.linalg.norm(reference, axis=1)

        rows.append(
            {
                "dt": dt,
                "max_relative_error_euclidean": float(euclidean_error.max()),
                "max_relative_error_transient": float(transient_error.max()),
                "final_relative_error_euclidean": float(euclidean_error[-1]),
                "final_relative_error_transient": float(transient_error[-1]),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "time_step_sensitivity.csv", index=False)

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(frame["dt"], frame["max_relative_error_euclidean"], label="Euclidean adaptive")
    plt.plot(frame["dt"], frame["max_relative_error_transient"], label="Transient-aware adaptive")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Time step $\Delta t$")
    plt.ylabel(r"Maximum relative $L^2$ error")
    plt.title("Time-step sensitivity")
    plt.legend()
    save_figure("time_step_sensitivity")

    return frame





# -----------------------------------------------------------------------------
# 8. Accuracy--resource and dimension studies
# -----------------------------------------------------------------------------
def run_budget_sweep(problem):
    """Compare Euclidean and transient-aware selection under equal generator caps."""
    rows = []
    generator = problem["L"]
    metric = transient_energy_matrix(generator, TAU_MAIN)

    for budget in BUDGET_GRID:
        euclidean = simulate_variational(
            generator,
            problem["y0"],
            DT,
            T_FINAL,
            BASE_PAIRS,
            mode="euclidean",
            transient_metric=metric,
            threshold=EUCLIDEAN_THRESHOLD,
            max_active_generators=budget,
        )
        transient = simulate_variational(
            generator,
            problem["y0"],
            DT,
            T_FINAL,
            BASE_PAIRS,
            mode="transient",
            transient_metric=metric,
            threshold=TRANSIENT_THRESHOLD,
            max_active_generators=budget,
        )
        reference = expm_multiply(
            generator,
            problem["y0"],
            start=0.0,
            stop=T_FINAL,
            num=int(round(T_FINAL / DT)) + 1,
            endpoint=True,
        )
        for method, result in [("Euclidean adaptive", euclidean), ("Transient-aware adaptive", transient)]:
            relative_error = np.linalg.norm(reference - result["y"], axis=1) / np.linalg.norm(reference, axis=1)
            rows.append(
                {
                    "method": method,
                    "max_active_generators": budget,
                    "max_relative_error": float(relative_error.max()),
                    "final_relative_error": float(relative_error[-1]),
                    "final_active_generators": int(result["active_count"][-1]),
                    "number_of_added_generators": len(result["selected_generators"]),
                    "time_of_last_enrichment": (
                        float(result["selected_generators"][-1]["time"])
                        if result["selected_generators"] else np.nan
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "budget_tradeoff.csv", index=False)

    plt.figure(figsize=(7.0, 4.5))
    for method, subset in frame.groupby("method"):
        subset = subset.sort_values("max_active_generators")
        plt.plot(subset["max_active_generators"], subset["max_relative_error"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("Maximum active generators")
    plt.ylabel(r"Maximum relative $L^2$ error")
    plt.title("Accuracy--resource trade-off")
    plt.legend()
    save_figure("budget_tradeoff")
    return frame


def run_dimension_study():
    """Algorithmic dimension/qubit scaling study, not a spatial convergence test."""
    rows = []

    for n_interior in DIMENSION_GRID:
        budget = max(len(BASE_PAIRS), int(round(DIMENSION_BUDGET_FRACTION * n_interior)))
        problem = build_problem(n_interior, A0, BETA, D0, GAMMA0, LAMBDA_BC)
        generator = problem["L"]
        metric = transient_energy_matrix(generator, TAU_MAIN)

        euclidean = simulate_variational(
            generator,
            problem["y0"],
            DT,
            T_FINAL,
            BASE_PAIRS,
            mode="euclidean",
            transient_metric=metric,
            threshold=EUCLIDEAN_THRESHOLD,
            max_active_generators=budget,
        )
        transient = simulate_variational(
            generator,
            problem["y0"],
            DT,
            T_FINAL,
            BASE_PAIRS,
            mode="transient",
            transient_metric=metric,
            threshold=TRANSIENT_THRESHOLD,
            max_active_generators=budget,
        )
        reference = expm_multiply(
            generator,
            problem["y0"],
            start=0.0,
            stop=T_FINAL,
            num=int(round(T_FINAL / DT)) + 1,
            endpoint=True,
        )
        spectral_abscissa = float(np.max(la.eigvals(generator).real))
        numerical_abscissa = float(np.max(la.eigvalsh((generator + generator.T) / 2.0)))
        nonnormality = float(la.norm(generator @ generator.T - generator.T @ generator, ord=2))

        for method, result in [("Euclidean adaptive", euclidean), ("Transient-aware adaptive", transient)]:
            relative_error = np.linalg.norm(reference - result["y"], axis=1) / np.linalg.norm(reference, axis=1)
            rows.append(
                {
                    "method": method,
                    "n_interior": n_interior,
                    "qubits": int(round(math.log2(n_interior))),
                    "max_active_generators": budget,
                    "relative_generator_budget": budget / n_interior,
                    "max_relative_error": float(relative_error.max()),
                    "final_relative_error": float(relative_error[-1]),
                    "final_active_generators": int(result["active_count"][-1]),
                    "spectral_abscissa": spectral_abscissa,
                    "numerical_abscissa": numerical_abscissa,
                    "nonnormality_measure": nonnormality,
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "dimension_study.csv", index=False)

    plt.figure(figsize=(7.0, 4.5))
    for method, subset in frame.groupby("method"):
        subset = subset.sort_values("n_interior")
        plt.plot(subset["n_interior"], subset["max_relative_error"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel(r"Interior dimension $n_h$")
    plt.ylabel(r"Maximum relative $L^2$ error")
    plt.title("Dimension and qubit-count study")
    plt.legend()
    save_figure("dimension_study")
    return frame




# -----------------------------------------------------------------------------
# 9. Execute all experiments and save metadata
# -----------------------------------------------------------------------------
def main():
    print(f"Saving outputs in: {OUTPUT_DIR.resolve()}")
    print("Running main benchmark ...")
    main_result = run_main_benchmark()

    print("\nMain benchmark summary:")
    print(main_result["summary"].to_string(index=False))
    print("\nNon-normality diagnostics:")
    print(f"alpha(L_h) = {main_result['spectral_abscissa']:.10f}")
    print(f"mu_2(L_h) = {main_result['numerical_abscissa']:.10f}")
    print(f"N_h        = {main_result['nonnormality']:.10f}")
    print(
        "\nCertificate check (sampled nodes):",
        "Euclidean =",
        bool(np.all(main_result["certificate_euclidean"] >= main_result["absolute_error_euclidean"] - 1.0e-10)),
        "; transient-aware =",
        bool(np.all(main_result["certificate_transient"] >= main_result["absolute_error_transient"] - 1.0e-10)),
    )

    if RUN_HORIZON_SWEEP:
        print("\nRunning horizon sensitivity ...")
        horizon_frame = run_horizon_sensitivity(main_result["problem"])
        print(horizon_frame.to_string(index=False))

    if RUN_BUDGET_SWEEP:
        print("\nRunning accuracy--resource budget sweep ...")
        budget_frame = run_budget_sweep(main_result["problem"])
        print(budget_frame.to_string(index=False))

    if RUN_NONNORMALITY_SWEEP:
        print("\nRunning non-normality sweep at a constrained resource budget ...")
        nonnormality_frame = run_nonnormality_sweep()
        print(nonnormality_frame.to_string(index=False))

    if RUN_TIME_STEP_SWEEP:
        print("\nRunning time-step sensitivity ...")
        timestep_frame = run_time_step_sensitivity(main_result["problem"])
        print(timestep_frame.to_string(index=False))

    if RUN_DIMENSION_STUDY:
        print("\nRunning dimension and qubit-count study ...")
        dimension_frame = run_dimension_study()
        print(dimension_frame.to_string(index=False))

    metadata = {
        "N_INTERIOR": N_INTERIOR,
        "A0": A0,
        "BETA": BETA,
        "D0": D0,
        "GAMMA0": GAMMA0,
        "DT": DT,
        "T_FINAL": T_FINAL,
        "TAU_MAIN": TAU_MAIN,
        "EUCLIDEAN_THRESHOLD": EUCLIDEAN_THRESHOLD,
        "TRANSIENT_THRESHOLD": TRANSIENT_THRESHOLD,
        "MAX_ACTIVE_GENERATORS": MAX_ACTIVE_GENERATORS,
        "LAMBDA_BC": LAMBDA_BC,
        "HORIZON_GRID": HORIZON_GRID,
        "BETA_GRID": BETA_GRID,
        "DT_GRID": DT_GRID,
        "BUDGET_GRID": BUDGET_GRID,
        "DIMENSION_GRID": DIMENSION_GRID,
        "CONSTRAINED_BUDGET": CONSTRAINED_BUDGET,
        "note": (
            "Euclidean McLachlan projection is used for time evolution. "
            "The finite-horizon metric is used only for transient-aware ansatz selection."
        ),
    }
    with open(DATA_DIR / "run_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nCompleted. Generated figures:")
    for file in sorted(FIG_DIR.glob("*.pdf")):
        print(" -", file.name)


if __name__ == "__main__":
    main()
