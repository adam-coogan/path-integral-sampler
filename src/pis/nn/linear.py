from typing import Callable, Optional, Tuple

import jax.random as jrandom
from equinox.module import Module, static_field
from jaxtyping import Array  # type: ignore

from .initializers import default_uniform_init


class Linear(Module):
    """Performs a linear transformation."""

    weight: Array
    bias: Optional[Array]
    in_features: int = static_field()
    out_features: int = static_field()
    use_bias: bool = static_field()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        init_fn: Callable[
            [jrandom.PRNGKeyArray, int, int, Tuple[int, ...]], Array
        ] = default_uniform_init,
        *,
        key: jrandom.PRNGKeyArray
    ):
        """**Arguments:**
        - `in_features`: The input size.
        - `out_features`: The output size.
        - `use_bias`: Whether to add on a bias as well.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        wkey, bkey = jrandom.split(key)
        self.weight = init_fn(
            wkey, in_features, out_features, (out_features, in_features)
        )
        if use_bias:
            self.bias = init_fn(bkey, in_features, out_features, (out_features,))

    def __call__(
        self, x: Array, *, key: Optional[jrandom.PRNGKeyArray] = None
    ) -> Array:
        """**Arguments:**
        - `x`: The input. Should be a JAX array of shape `(in_features,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        !!! info
            If you want to use higher order tensors as inputs (for example featuring batch dimensions) then use
            `jax.vmap`. For example, for an input `x` of shape `(batch, in_features)`, using
            ```python
            linear = equinox.nn.Linear(...)
            jax.vmap(linear)(x)
            ```
            will produce the appropriate output of shape `(batch, out_features)`.
        **Returns:**
        A JAX array of shape `(out_features,)`
        """
        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        return x
