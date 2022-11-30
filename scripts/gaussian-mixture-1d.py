from jax.config import config  # type: ignore

config.update("jax_debug_nans", True)

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist
import optax
from jax.random import PRNGKey, split
from tqdm.auto import trange

from pis import PathIntegralSampler
from pis.nn import ControlNet

plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["text.usetex"] = True

X_DIM = 1


def get_log_mu(x):
    component_dist = dist.Normal(jnp.array((-5.0, 5.0)))
    return jnp.log(jnp.exp(component_dist.log_prob(x)).sum(0)) + jnp.log(0.5)


def sample(key, pis, model, n_samples):
    key, *subkeys = split(key, n_samples + 1)
    subkeys = jnp.stack(subkeys)
    xs, log_ws = jax.vmap(pis.get_sample, (None, 0, None))(model, subkeys, get_log_mu)
    return xs, log_ws


def plot(losses, xs, log_ws):
    fig = plt.figure(figsize=(12, 3))

    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 2)
    plt.hist(
        xs[:, 0],
        bins=50,
        weights=jnp.exp(log_ws),
        density=True,
        histtype="step",
        label="Samples",
    )
    x_grid = jnp.linspace(-9, 9, 400)
    plt.plot(x_grid, jnp.exp(jax.vmap(get_log_mu)(x_grid)), label="Terminal density")
    plt.legend(frameon=False)
    plt.xlabel(r"$x$")
    plt.ylabel("Density")

    plt.subplot(1, 3, 3)
    plt.hist(log_ws, bins=50)
    plt.xlabel(r"$\log w$")
    plt.ylabel("Count")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    get_score_mu = jax.grad(get_log_mu)

    # Set up sampler
    t0 = 0.0
    t1 = 1.0
    dt0 = 0.05
    pis = PathIntegralSampler(X_DIM, t1, dt0)

    key = PRNGKey(86)

    # Construct the network
    key, subkey = split(key)
    model = ControlNet(subkey, X_DIM, get_score_mu, 64, 3, T=t1)

    lr = 1e-3
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))  # type: ignore
    batch_size = 256
    n_steps = 500
    losses = []
    print("training")
    with trange(n_steps) as pbar:
        for _ in pbar:
            key, subkey = split(key)
            loss, model, opt_state = eqx.filter_jit(pis.train_step)(
                model, subkey, get_log_mu, batch_size, opt_state, optim
            )
            losses.append(loss.item())
            pbar.set_postfix(loss=losses[-1])

    losses = jnp.array(losses)

    n_samples = 20000
    print(f"drawing {n_samples} samples")
    key, subkey = split(key)
    xs, log_ws = sample(subkey, pis, model, n_samples)

    print("plotting")
    fig = plot(losses, xs, log_ws)
    fig.savefig("output/gaussian-mixture-1d.png")

    print("done!")
