import jax.numpy as jnp
import numpy.random as random


def wavefunction_after_measurement(psi: jnp.ndarray, site: int):
    """
    Computes the wavefunction after a measurement.

    Parameters:
    psi (jnp.ndarray): The initial state vector.
    site (int): The measurement outcome.

    Returns:
    jnp.ndarray: The state vector after the measurement.
    """
    # key = jax.random.key(42)

    site -= 1
    # phi = jnp.zeros(len(psi))
    p = jnp.abs(psi[site]) ** 2
    if p < 1e-10:
        p = 0

    r = random.rand()
    if r <= p:
        phi = jnp.zeros(len(psi))
        phi = phi.at[site].set(1)
    else:
        phi = psi.copy()
        phi = phi.at[site].set(0)
        phi = phi / jnp.linalg.norm(phi)

    return phi
