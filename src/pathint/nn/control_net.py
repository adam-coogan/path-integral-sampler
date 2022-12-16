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
    architecture close to the one from the `path integral sampler paper <https://arxiv.org/abs/2111.15141>`_.
    This is of the form :math:`a_\\theta(t, x) + b_\\theta(t, x) \\nabla \\log \\mu(x)`,
    with the output initialized to zero using an overall learn multiplicative factor.
    """

    get_score_mu: Callable[[Array], Array] = eqx.static_field()
    T: float = eqx.static_field()
    t_pos_encoding: Callable
    t_emb: Callable
    x_emb: Callable
    const_net: Callable
    coeff_net: Callable
    output_scaling: Array

    def __init__(
        self,
        x_dim: int,
        get_score_mu: Callable[[Array], Array],
        T: float = 1.0,
        L_max: int = 32,
        emb_dim: int = 64,
        embed_width_size: int = 64,
        embed_depth: int = 2,
        width_size: int = 64,
        depth: int = 3,
        output_scaling: Array = jnp.array(0.0),
        scalar_coeff_net: bool = True,
        act: Callable[[Array], Array] = jax.nn.relu,
        weight_init=default_uniform_init,
        bias_init=default_uniform_init,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            x_dim: size of :math:`x` vector.
            get_score_mu: score of the target density, :math:`\\nabla \\log \\mu(x)`.
            T: duration of diffusion.
            L_max: :math:`L` parameter for positional encoding of :math:`t`.
            emb_dim: dimension for embedding of :math:`t` and :math:`x`.
            embed_width_size: hidden layer dimensionality of embedding networks.
            embed_depth: depth of embedding networks.
            width_size: hidden layer dimensionality for MLPs mapping embeddings to
                :math:`a_\\theta(t, x)` and :math:`b_\\theta(t, x)`.
            depth: depth of MLPs mapping embeddings to :math:`a_\\theta(t, x)`
                and :math:`b_\\theta(t, x)`.
            output_scaling: initial scaling of both networks.
            scalar_coeff_net: if `True`, :math:`b_\\theta(t, x)` will output a scalar
                instead of a vector to be multiplied elementwise with :math:`\\nabla \\log \\mu(x)`.
            act: activation used in all networks.
            weight_init: function for initializing weights.
            bias_init: function for initializing biases.
            key: PRNG key for initializing layers.
        """
        super().__init__()
        self.get_score_mu = get_score_mu
        self.T = T
        self.output_scaling = output_scaling

        # Build layers
        key_t, key_x, key_const, key_coeff = split(key, 4)
        self.t_pos_encoding = PositionalEncoding(L_max)
        self.t_emb = eqx.nn.MLP(
            2 * L_max,
            emb_dim,
            embed_width_size,
            embed_depth,
            activation=act,
            key=key_t,
        )
        self.x_emb = eqx.nn.MLP(
            x_dim,
            emb_dim,
            embed_width_size,
            embed_depth,
            activation=act,
            key=key_x,
        )
        self.const_net = eqx.nn.MLP(
            emb_dim,
            x_dim,
            width_size,
            depth,
            activation=act,
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
            activation=act,
            key=key_coeff,
        )

        # Reinitialize weights
        self.t_emb = apply_linear_init(
            key_t, weight_init, bias_init, self.t_emb
        )
        self.x_emb = apply_linear_init(
            key_x, weight_init, bias_init, self.x_emb
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
        t_emb = self.t_emb(t_emb)

        # Normalize to Gaussian sample for uncontrolled process
        x_norm = x / jnp.sqrt(self.T)
        x_emb = self.x_emb(x_norm)
        tx_emb = t_emb + x_emb

        const = self.const_net(tx_emb)
        coeff = self.coeff_net(tx_emb)

        score = self.get_score_mu(x)

        return self.output_scaling * (const + coeff * score)
