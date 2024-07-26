from jax import jit
import jax.numpy as jnp


@jit
def probability_distribution(psi):
    return jnp.abs(psi) ** 2
