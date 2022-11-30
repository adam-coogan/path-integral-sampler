import math
from typing import Tuple

import jax.numpy as jnp
from jax.random import PRNGKeyArray, normal, uniform
from jaxtyping import Array  # type: ignore


def default_uniform_init(
    key: PRNGKeyArray, in_features: int, out_features: int, shape: Tuple[int, ...]
) -> Array:
    lim = 1 / math.sqrt(in_features)
    return uniform(key, shape, minval=-lim, maxval=lim)


def lecun_normal(
    key: PRNGKeyArray, in_features: int, out_features: int, shape: Tuple[int, ...]
) -> Array:
    scale = 1 / jnp.sqrt(in_features)
    return normal(key, shape) * scale
