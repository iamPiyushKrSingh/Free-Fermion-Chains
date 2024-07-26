import jax.numpy as jnp
from scipy.constants import golden_ratio
from jax import jit
from functools import partial


@partial(jit, static_argnames=("L", "periodic"))
def aubrey_andre_hamiltonian(
    L: jnp.int64,
    J: jnp.float64,
    λ: jnp.float64,
    β: jnp.float64 = golden_ratio,
    φ: jnp.float64 = 0.0,
    periodic: bool = False,
) -> jnp.ndarray:
    """
    Generates the Aubry-André Hamiltonian matrix.

    Parameters:
    L (int): The size of the system (number of lattice sites).
    J (float): Hopping amplitude between nearest-neighbor sites.
    λ (float): Strength of the quasiperiodic potential.
    β (float): Irrational number (commonly taken as the inverse of the golden ratio).
    φ (float): Phase parameter of the potential.

    Returns:
    jnp.ndarray: The Hamiltonian matrix of size LxL.
    """
    # Generate the diagonal elements
    diag = λ * jnp.cos(2 * jnp.pi * β * jnp.arange(1, L + 1) + φ)
    # Generate the off-diagonal elements
    off_diag = -J * jnp.ones(L - 1)
    # Generate the Hamiltonian
    H = jnp.diag(diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)

    if periodic:
        H = H.at[0, -1].set(-J)
        H = H.at[-1, 0].set(-J)

    return H
