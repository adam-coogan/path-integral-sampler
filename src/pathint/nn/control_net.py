from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.random import PRNGKeyArray, split
from jaxtyping import Array  # type: ignore

# from .composed import MLP
from .init import apply_linear_init, default_uniform_init
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
        weight_init=default_uniform_init,
        bias_init=default_uniform_init,
    ):
        super().__init__()
        self.get_score = get_score
        self.T = T
        self.output_scaling = output_scaling

        # Build layers
        key_t, key_x, key_const, key_coeff = split(key, 4)
        self.t_pos_encoding = PositionalEncoding(L_max)
        self.t_embedding_net = eqx.nn.MLP(
            2 * L_max,
            emb_dim,
            embed_width_size,
            embed_depth,
            activation=activation,
            key=key_t,
        )
        self.x_embedding_net = eqx.nn.MLP(
            x_dim,
            emb_dim,
            embed_width_size,
            embed_depth,
            activation=activation,
            key=key_x,
        )
        self.const_net = eqx.nn.MLP(
            emb_dim,
            x_dim,
            width_size,
            depth,
            activation=activation,
            key=key_const,
        )
        self.const_net = apply_linear_init(
            key_const, weight_init, bias_init, self.const_net
        )
        coeff_net_out_size = 1 if scalar_coeff_net else x_dim
        self.coeff_net = eqx.nn.MLP(
            emb_dim,
            coeff_net_out_size,
            width_size,
            depth,
            activation=activation,
            key=key_coeff,
        )

        # Reinitialize weights
        self.t_embedding_net = apply_linear_init(
            key_t, weight_init, bias_init, self.t_embedding_net
        )
        self.x_embedding_net = apply_linear_init(
            key_x, weight_init, bias_init, self.x_embedding_net
        )
        self.const_net = apply_linear_init(
            key_const, weight_init, bias_init, self.const_net
        )
        self.coeff_net = apply_linear_init(
            key_coeff, weight_init, bias_init, self.coeff_net
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
