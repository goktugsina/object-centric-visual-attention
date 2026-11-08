"""Small conditional choice model; no pretrained network or image input."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, softmax


def objective(theta, features, targets, weights, l2):
    loss = .5 * l2 * np.dot(theta, theta)
    gradient = l2 * theta.copy()
    for x, q, weight in zip(features, targets, weights):
        z = x @ theta
        p = softmax(z)
        loss += weight * (logsumexp(z) - q @ z)
        gradient += weight * (x.T @ (p - q))
    return float(loss), gradient


def packed_objective(theta, x, starts, row_weights, target_weights, weights, l2):
    """The same image-balanced objective with variable-size segments packed."""
    z = x @ theta
    sizes = np.diff(np.r_[starts, len(z)])
    maxima = np.maximum.reduceat(z, starts)
    exponentials = np.exp(z - np.repeat(maxima, sizes))
    totals = np.add.reduceat(exponentials, starts)
    p = exponentials / np.repeat(totals, sizes)
    loss = weights @ (maxima + np.log(totals)) - target_weights @ z + .5*l2*(theta@theta)
    gradient = x.T @ (row_weights*p-target_weights) + l2*theta
    return float(loss), gradient


def fit(scenes, n_features, l2=1e-3, multiplicities=None):
    xs = [s["features"][:, :n_features] for s in scenes]
    qs = [s["counts"] / s["counts"].sum() for s in scenes]
    weights = np.ones(len(scenes)) if multiplicities is None else np.asarray(multiplicities, float)
    weights /= weights.sum()
    sizes = np.array([len(x) for x in xs])
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    row_weights = np.repeat(weights, sizes)
    result = minimize(packed_objective, np.zeros(n_features),
                      args=(np.vstack(xs), starts, row_weights, np.concatenate(qs)*row_weights, weights, l2),
                      method="L-BFGS-B", jac=True,
                      options=dict(maxiter=1000, ftol=1e-11, gtol=1e-7))
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result.x


def predict(scenes, theta, temperature=1.0):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return [softmax(s["features"][:, :len(theta)] @ theta / temperature) for s in scenes]


def temperature_scale(probabilities, temperature):
    return [softmax(np.log(np.maximum(p, 1e-300)) / temperature) for p in probabilities]


def fit_temperature(scenes, probabilities):
    qs = [s["counts"] / s["counts"].sum() for s in scenes]
    def loss(log_temperature):
        ps = temperature_scale(probabilities, np.exp(log_temperature))
        return np.mean([-q @ np.log(p) for q, p in zip(qs, ps)])
    result = minimize_scalar(loss, bounds=(np.log(.25), np.log(4)), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature fitting failed")
    # T=1 is explicitly included so numerical optimization cannot worsen val NLL.
    return float(np.exp(result.x)) if result.fun < loss(0.) else 1.
