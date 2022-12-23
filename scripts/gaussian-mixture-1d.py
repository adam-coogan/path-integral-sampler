from jax.config import config  # type: ignore

config.update("jax_debug_nans", True)

from warnings import filterwarnings

import click
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist
import optax
from jax.random import PRNGKey, split
from jaxtyping import Array  # type: ignore
from tqdm.auto import trange

from pathint import PathIntegralSampler
from pathint.nn import ControlNet
from pathint.nn.init import lecun_init, zeros_init

filterwarnings("ignore", module="diffrax.integrate", category=FutureWarning)
plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["text.usetex"] = True

X_SIZE = 1


def get_log_mu(x):
    component_dist = dist.Normal(jnp.array((-5.0, 5.0)))
    return jnp.log(jnp.exp(component_dist.log_prob(x)).sum(0)) + jnp.log(0.5)


def sample(key, pathint, model, n_samples):
    key, *subkeys = split(key, n_samples + 1)
    subkeys = jnp.stack(subkeys)
    xs, log_ws = jax.vmap(pathint.get_sample, (None, 0))(model, subkeys)
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


get_score_mu = jax.grad(get_log_mu)


@click.command()
@click.option("--seed", default=20, help="PRNG seed")
@click.option("--n-steps", default=200, help="PRNG seed")
def main(seed, n_steps):
    key = PRNGKey(seed)

    # Set up sampler
    t1 = 10.0
    dt0 = t1 / 100
    pathint = PathIntegralSampler(get_log_mu, X_SIZE, t1, dt0)

    # Construct the network
    key, subkey = split(key)
    model = ControlNet(X_SIZE, lambda _, x: get_score_mu(x), t1, key=subkey)
    lr = 3e-4
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))  # type: ignore
    batch_size = 128
    loss_fn = lambda model, key: jax.vmap(pathint.get_loss, (None, 0))(
        model, key
    ).mean()

    @eqx.filter_jit
    def train_step(model, opt_state, key, batch_size):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, split(key, batch_size))
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    losses = []
    print("training")
    with trange(n_steps) as pbar:
        for _ in pbar:
            key, subkey = split(key)
            model, opt_state, loss = train_step(model, opt_state, subkey, batch_size)
            losses.append(loss.item())
            pbar.set_postfix(loss=losses[-1])

    losses = jnp.array(losses)

    n_samples = 10_000
    print(f"drawing {n_samples} samples")
    key, subkey = split(key)
    xs, log_ws = sample(subkey, pathint, model, n_samples)

    print("plotting")
    fig = plot(losses, xs, log_ws)
    fig.savefig("output/gaussian-mixture-1d.png")

    print("done!")


if __name__ == "__main__":
    main()
