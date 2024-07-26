import jax.numpy as jnp
from jax import jit, random
from functools import partial


@partial(jit, static_argnames=("L", "periodic"))
def anderson_hamiltonian(
    L: jnp.int64,
    J: jnp.float64,
    W: jnp.float64,
    seed: jnp.int64 = 42,
    periodic: bool = False,
) -> jnp.ndarray:
    """
    Generate a Anderson model Hamiltonian.

    Parameters
    ----------
    L : jnp.int64
        The number of sites.
    J : jnp.float64
        The hopping parameter.
    W : jnp.float64
        The disorder strength.
    seed : jnp.int64
        The seed for the random number generator.

    Returns
    -------
    H : jnp.ndarray
        The Hamiltonian.
    """
    μ = 0.0
    σ = W / jnp.sqrt(3)
    key = random.PRNGKey(seed)

    diag = μ + σ * random.normal(key, shape=(L,))

    H = jnp.diag(diag) + J * (
        jnp.diag(jnp.ones(L - 1), k=1) + jnp.diag(jnp.ones(L - 1), k=-1)
    )

    if periodic:
        H = H.at[0, -1].set(J)
        H = H.at[-1, 0].set(J)

    return H
