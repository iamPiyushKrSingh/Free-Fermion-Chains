import jax.numpy as jnp
import numpy as np

from ..utils import (
    finer_time_evolutions,
    probability_distribution,
    wavefunction_after_measurement,
)


def spacetime_probability_distribution_finer_disorder(
    H: jnp.ndarray,
    psi: jnp.ndarray,
    Δt: jnp.float64,
    n: jnp.int64,
    generations: jnp.int64,
) -> np.ndarray:
    """
    This function calculates the probability distribution of the system in space and time.
    """
    probabilities = []
    probabilities.append(probability_distribution(psi))
    for _ in range(generations):
        psi, prob = finer_time_evolutions(H, Δt, n, psi)
        probabilities.extend(prob)
        psi = wavefunction_after_measurement(psi, np.random.randint(1, len(psi) + 1))
        probabilities.append(probability_distribution(psi))

    return np.array(probabilities)


def spacetime_probability_distribution_finer_disorder_path(
    H: jnp.ndarray,
    psi: jnp.ndarray,
    Δt: jnp.float64,
    n: jnp.int64,
    path: np.ndarray,
    generations: jnp.int64,
) -> np.ndarray:
    """
    Calculate the spacetime probability distribution for a finer disorder path.

    Args:
        H (jnp.ndarray): The Hamiltonian matrix.
        psi (jnp.ndarray): The initial wavefunction.
        Δt (jnp.float64): The time step.
        n (jnp.int64): The number of time steps.
        path (np.ndarray): The disorder path.
        generations (jnp.int64): The number of generations.

    Returns:
        np.ndarray: The spacetime probability distribution.

    """
    probabilities = []
    probabilities.append(probability_distribution(psi))
    for i in range(generations):
        psi, prob = finer_time_evolutions(H, Δt, n, psi)
        probabilities.extend(prob)
        psi = wavefunction_after_measurement(psi, path[i])
        probabilities.append(probability_distribution(psi))

    return np.array(probabilities)
