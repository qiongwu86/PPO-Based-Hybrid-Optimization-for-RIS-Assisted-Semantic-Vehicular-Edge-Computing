"""
Lambda LP solver for minimizing maximum branch delay with fixed k (semantic version).

Problem per task:
  minimize    t
  subject to  A_local * lambda_local <= t
              A_edge  * lambda_edge  <= t
              A_bs    * lambda_bs    <= t
              lambda_local + lambda_edge + lambda_bs = 1
              lambda_* >= 0

Closed-form optimum exists:
  t* = 1 / (1/A_local + 1/A_edge + 1/A_bs)
  lambda_j* = t* / A_j

If SciPy is available, we also provide a linprog-based solver for validation.
"""

from typing import Tuple


def _closed_form_solution(A_local: float, A_edge: float, A_bs: float) -> Tuple[float, float, float, float]:
    eps = 1e-12
    A_local = float(max(A_local, eps))
    A_edge = float(max(A_edge, eps))
    A_bs = float(max(A_bs, eps))

    inv_sum = (1.0 / A_local) + (1.0 / A_edge) + (1.0 / A_bs)
    t_star = 1.0 / inv_sum
    lambda_local = t_star / A_local
    lambda_edge = t_star / A_edge
    lambda_bs = t_star / A_bs

    # Numerical guard: normalize to ensure sum = 1 exactly
    s = lambda_local + lambda_edge + lambda_bs
    if s <= eps:
        # Fallback: allocate all to the smallest A_j (min delay per unit lambda)
        A_values = [A_local, A_edge, A_bs]
        idx_min = A_values.index(min(A_values))
        lambdas = [0.0, 0.0, 0.0]
        lambdas[idx_min] = 1.0
        t_val = A_values[idx_min]
        return lambdas[0], lambdas[1], lambdas[2], t_val

    lambda_local /= s
    lambda_edge /= s
    lambda_bs /= s
    t_star = max(A_local * lambda_local, A_edge * lambda_edge, A_bs * lambda_bs)
    return lambda_local, lambda_edge, lambda_bs, t_star


def _linprog_solution(A_local: float, A_edge: float, A_bs: float) -> Tuple[float, float, float, float]:
    try:
        from scipy.optimize import linprog
    except Exception:
        return _closed_form_solution(A_local, A_edge, A_bs)

    # Variables: x = [lambda_local, lambda_edge, lambda_bs, t]
    c = [0.0, 0.0, 0.0, 1.0]
    A_ub = [
        [A_local, 0.0, 0.0, -1.0],
        [0.0, A_edge, 0.0, -1.0],
        [0.0, 0.0, A_bs, -1.0],
    ]
    b_ub = [0.0, 0.0, 0.0]
    A_eq = [[1.0, 1.0, 1.0, 0.0]]
    b_eq = [1.0]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, None)]

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success or res.x is None:
        return _closed_form_solution(A_local, A_edge, A_bs)

    lambda_local, lambda_edge, lambda_bs, t_val = res.x.tolist()
    # Numerical guard: clip and renormalize
    lambda_local = max(0.0, min(1.0, lambda_local))
    lambda_edge = max(0.0, min(1.0, lambda_edge))
    lambda_bs = max(0.0, min(1.0, lambda_bs))
    s = lambda_local + lambda_edge + lambda_bs
    if s > 0:
        lambda_local /= s
        lambda_edge /= s
        lambda_bs /= s
    t_val = max(A_local * lambda_local, A_edge * lambda_edge, A_bs * lambda_bs)
    return lambda_local, lambda_edge, lambda_bs, t_val


def compute_optimal_lambda(A_local: float, A_edge: float, A_bs: float, prefer: str = "closed_form") -> Tuple[float, float, float, float]:
    """
    Compute optimal (lambda_local, lambda_edge, lambda_bs, t_opt).

    prefer: "closed_form" | "linprog" | "auto"
    """
    if prefer == "linprog":
        return _linprog_solution(A_local, A_edge, A_bs)
    if prefer == "auto":
        # Try linprog first, fall back to closed form
        try:
            return _linprog_solution(A_local, A_edge, A_bs)
        except Exception:
            return _closed_form_solution(A_local, A_edge, A_bs)
    # Default closed-form
    return _closed_form_solution(A_local, A_edge, A_bs)


