from functools import partial

from jax import jit
import jax.numpy as jnp
import jax.scipy as jsp

from .probability_distribution import probability_distribution


@partial(jit, static_argnames=("n"))
def finer_time_evolutions(
    H: jnp.ndarray, Δt: jnp.float64, n: jnp.int64, psi: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    This function calculates the time evolution of the system for a given Hamiltonian.
    """
    dt = Δt / n
    udt = jsp.linalg.expm(-1j * H * dt)
    prob = []
    for _ in range(n):
        psi = jnp.matmul(udt, psi)
        prob.append(probability_distribution(psi))

    prob = jnp.asarray(prob)
    return psi, prob
