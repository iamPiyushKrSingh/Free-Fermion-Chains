import jax.numpy as jnp
import numpy as np

from ..utils import (
    time_evolution,
    probability_distribution,
    wavefunction_after_measurement,
)


def spacetime_probability_distribution(
    H: jnp.ndarray,
    psi: jnp.ndarray,
    Δt: jnp.float64,
    n: jnp.int64,
    generations: jnp.int64,
) -> np.ndarray:
    """
    This function calculates the probability distribution of the system in space and time.
    """
    # Udt = jsp.linalg.expm(-1j * H * Δt/n)
    probabilities = []
    probabilities.append(probability_distribution(psi))
    for _ in range(generations):
        psi = time_evolution(H, psi, Δt, n)
        psi = wavefunction_after_measurement(psi, np.random.randint(1, len(psi) + 1))
        probabilities.append(probability_distribution(psi))

    return np.array(probabilities)
