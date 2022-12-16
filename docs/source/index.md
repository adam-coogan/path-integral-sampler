# Welcome to `pathint`'s documentation!

`pathint` is a jax implementation of the [path integral sampler](https://arxiv.org/abs/2111.15141),
a method based on the Schrödinger bridge problem for sampling from (unnormalized)
probability densities. Behind the scenes it relies on [diffrax](https://github.com/patrick-kidger/diffrax)
to handle the stochastic differential equations, [equinox](https://github.com/patrick-kidger/equinox)
for the neural networks and [optax](https://github.com/deepmind/optax/) for training.

Check out the links below to learn more.

```{eval-rst}
.. toctree::
   :maxdepth: 1

   api
```

