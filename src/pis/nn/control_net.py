from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.random import PRNGKeyArray, split
from jaxtyping import Array  # type: ignore

from .composed import MLP
from .initializers import default_uniform_init
from .positional_encoding import PositionalEncoding


class ControlNet(eqx.Module):
    """
    Affine transformation of score parametrized by two neural networks, using the
    same architecture as in the path integral sampler paper.
    """

    t_pos_encoding: Callable
    t_embedding_net: Callable
    x_embedding_net: Callable
    const_net: Callable
    coeff_net: Callable
    get_score: Callable[[Array], Array] = eqx.static_field()
    T: float = eqx.static_field()
    output_scaling: Array

    def __init__(
        self,
        key: PRNGKeyArray,
        x_dim: int,
        get_score: Callable[[Array], Array],
        width_size: int = 64,
        depth: int = 3,
        embed_width_size: int = 64,
        embed_depth: int = 2,
        T: float = 1.0,
        L_max: int = 32,
        emb_dim: int = 64,
        output_scaling: Array = jnp.array(0.03),
        scalar_coeff_net: bool = True,
        activation: Callable[[Array], Array] = jax.nn.relu,
        init_fn: Callable[
            [PRNGKeyArray, int, int, Tuple[int, ...]], Array
        ] = default_uniform_init,
    ):
        super().__init__()
        self.get_score = get_score
        self.T = T
        self.output_scaling = output_scaling

        key_t, key_const, key_coeff = split(key, 3)
        self.t_pos_encoding = PositionalEncoding(L_max)
        self.t_embedding_net = MLP(
            2 * L_max,
            emb_dim,
            embed_width_size,
            embed_depth,
            activation=activation,
            init_fn=init_fn,
            key=key_t,
        )
        self.x_embedding_net = MLP(
            x_dim,
            emb_dim,
            embed_width_size,
            embed_depth,
            init_fn=init_fn,
            activation=activation,
            key=key_t,
        )
        self.const_net = MLP(
            emb_dim,
            x_dim,
            width_size,
            depth,
            activation=activation,
            init_fn=init_fn,
            key=key_const,
        )
        coeff_net_out_size = 1 if scalar_coeff_net else x_dim
        self.coeff_net = MLP(
            emb_dim,
            coeff_net_out_size,
            width_size,
            depth,
            init_fn=init_fn,
            activation=activation,
            key=key_coeff,
        )

    def __call__(self, t: Array, x: Array) -> Array:
        t_emb = t / self.T - 0.5
        t_emb = self.t_pos_encoding(t_emb)
        t_emb = self.t_embedding_net(t_emb)

        # Normalize to Gaussian sample for uncontrolled process
        x_norm = x / jnp.sqrt(self.T)
        x_emb = self.x_embedding_net(x_norm)
        tx_emb = t_emb + x_emb

        const = self.const_net(tx_emb)
        coeff = self.coeff_net(tx_emb)

        score = self.get_score(x)

        return self.output_scaling * (const + coeff * score)
