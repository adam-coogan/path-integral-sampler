import math
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.random import PRNGKeyArray, normal, split, uniform
from jaxtyping import Array, PyTree  # type: ignore


def default_uniform_init(
    key: PRNGKeyArray, out_features: int, in_features: int, shape: Tuple[int, ...]
) -> Array:
    lim = 1 / math.sqrt(in_features)
    return uniform(key, shape, minval=-lim, maxval=lim)


def lecun_init(
    key: PRNGKeyArray, out_features: int, in_features: int, shape: Tuple[int, ...]
) -> Array:
    stdev = 1 / math.sqrt(in_features)
    return stdev * normal(key, shape)


def zeros_init(
    key: PRNGKeyArray, out_features: int, in_features: int, shape: Tuple[int, ...]
) -> Array:
    return jnp.zeros(shape)


def apply_linear_init(key, weight_init, bias_init, model: PyTree):
    """
    Reinitializes all linear layers in a model.
    """
    key_w, key_b = split(key)

    # Extract parameters
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    # The if is because not everything returned by tree_leaves is linear...
    get_weights = lambda m: [
        x.weight for x in jtu.tree_leaves(m, is_leaf=is_linear) if is_linear(x)
    ]
    get_biases = lambda m: [
        x.bias for x in jtu.tree_leaves(m, is_leaf=is_linear) if is_linear(x)
    ]
    weights = get_weights(model)
    out_features = [w.shape[0] for w in weights]
    in_features = [w.shape[0] for w in weights]
    biases = get_biases(model)

    # Reinitialize
    keys_w = split(key_w, len(weights))
    new_weights = [
        weight_init(k, o, i, w.shape)
        for k, o, i, w in zip(keys_w, out_features, in_features, weights)
    ]
    keys_b = split(key_b, len(biases))
    new_biases = [
        bias_init(k, o, i, b.shape)
        for k, o, i, b in zip(keys_b, out_features, in_features, biases)
    ]

    # Reconstitute model
    new_model = eqx.tree_at(get_weights, model, new_weights)
    new_model = eqx.tree_at(get_biases, new_model, new_biases)

    return new_model
