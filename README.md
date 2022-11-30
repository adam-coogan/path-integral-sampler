# path-integral-sampler

[![docs](https://readthedocs.org/projects/path-integral-sampler/badge/?version=latest)](http://path-integral-sampler.readthedocs.io/?badge=latest)

`pis` is a jax implementation of the [path integral sampler](https://arxiv.org/abs/2111.15141),
a method based on the Schrödinger bridge problem for sampling from (unnormalized)
probability densities. Behind the scenes it relies on [diffrax](https://github.com/patrick-kidger/diffrax)
to handle the stochastic differential equations, [equinox](https://github.com/patrick-kidger/equinox)
for the neural networks and [optax](https://github.com/deepmind/optax/) for training.

Check out the [docs](https://path-integral-sampler.readthedocs.io/en/latest/) badge
above for more details, or try out the scripts applying the method to some low-dimensional
problems (runnable on a laptop).
