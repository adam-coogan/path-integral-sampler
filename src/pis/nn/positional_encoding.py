import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array  # type: ignore
from octopus.problems.base import Array


class PositionalEncoding(eqx.Module):
    L_max: int = eqx.static_field()
    omegas: Array = eqx.static_field()

    def __init__(self, L_max: int):
        super().__init__()
        self.L_max = L_max
        self.omegas = 2 ** jnp.linspace(0, L_max - 1, L_max)

    def __call__(self, x: Array, *args, **kwargs) -> Array:
        angles = (self.omegas[None, :] * jnp.atleast_1d(x)[:, None]).flatten()
        return jnp.append(jnp.sin(angles), jnp.cos(angles))
