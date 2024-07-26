from functools import partial

from jax import jit
import jax.numpy as jnp
import jax.scipy as jsp


@partial(jit, static_argnames=("n_steps"))
def time_evolution(
    H: jnp.ndarray, psi: jnp.ndarray, t: jnp.float64, n_steps: int = 100
):
    """
    Time-evolves a state under a given Hamiltonian.

    Parameters:
    H (jnp.ndarray): The Hamiltonian matrix.
    psi (jnp.ndarray): The initial state vector.
    t (float): The time to evolve the state to.
    n_steps (int): The number of steps to use in the Trotter-Suzuki approximation.

    Returns:
    jnp.ndarray: The state vector evolved to time t.
    """
    dt = t / n_steps

    Udt = jsp.linalg.expm(-1j * H * dt)

    U = jnp.linalg.matrix_power(Udt, n_steps)

    return jnp.linalg.matmul(U, psi)
