import typing
from typing import Callable, List, Optional, Tuple

import jax.nn as jnn
from equinox.module import Module, static_field
from jax.random import PRNGKeyArray, split
from jaxtyping import Array  # type: ignore

from .initializers import default_uniform_init
from .linear import Linear


def _identity(x):
    return x


if getattr(typing, "GENERATING_DOCUMENTATION", False):

    def relu(_):
        pass

    jnn.relu = relu
    _identity.__qualname__ = "identity"  # Renamed for nicer documentation.


class MLP(Module):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network."""

    layers: List[Linear]
    activation: Callable
    final_activation: Callable
    in_size: int = static_field()
    out_size: int = static_field()
    width_size: int = static_field()
    depth: int = static_field()

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable = jnn.relu,
        final_activation: Callable = _identity,
        use_bias: bool = True,
        init_fn: Callable[
            [PRNGKeyArray, int, int, Tuple[int, ...]], Array
        ] = default_uniform_init,
        *,
        key: PRNGKeyArray,
        **kwargs,
    ):
        """**Arguments**:
        - `in_size`: The size of the input layer.
        - `out_size`: The size of the output layer.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers.
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """

        super().__init__(**kwargs)
        keys = split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(Linear(in_size, out_size, use_bias, init_fn, key=keys[0]))
        else:
            layers.append(Linear(in_size, width_size, use_bias, init_fn, key=keys[0]))
            for i in range(depth - 1):
                layers.append(
                    Linear(width_size, width_size, use_bias, init_fn, key=keys[i + 1])
                )
            layers.append(Linear(width_size, out_size, use_bias, init_fn, key=keys[-1]))
        self.layers = layers
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**
        - `x`: A JAX array with shape `(in_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        **Returns:**
        A JAX array with shape `(out_size,)`.
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x
