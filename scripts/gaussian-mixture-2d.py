from jax.config import config  # type: ignore

config.update("jax_debug_nans", True)

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist
import optax
from jax.random import PRNGKey, split
from pis import PathIntegralSampler
from pis.nn import ControlNet
from pis.nn.init import lecun_init, zeros_init
from tqdm.auto import trange

plt.rcParams["figure.facecolor"] = "w"
plt.rcParams["text.usetex"] = True


def sample(key, pis, model, n_samples):
    key, *subkeys = split(key, n_samples + 1)
    subkeys = jnp.stack(subkeys)
    xs, log_ws = jax.vmap(jax.jit(lambda key: pis.get_sample(model, key, get_log_mu)))(
        subkeys
    )
    return xs, log_ws


X_DIM = 2


def get_log_mu(x):
    clip_range = 9.0
    width = 1.0
    mu_xs = 1.7 * jnp.array((-3.0, -2.8, -3.2, 0.0, 0.2, 0.3, 2.9, 3.0, 2.7))
    mu_ys = 1.7 * jnp.array((-2.7, -0.4, 3.0, -2.9, 0.4, 2.6, -2.8, 0.1, 3.2))

    x_clipped = jnp.clip(x, -clip_range, clip_range)
    mus = jnp.stack((mu_xs, mu_ys), -1)
    log_probs = jax.vmap(lambda mu: dist.Normal(mu, width).log_prob(x_clipped))(mus)
    return jnp.log(jnp.exp(log_probs.sum(-1)).sum(0)) + jnp.log(1 / len(mus))


get_score_mu = jax.grad(get_log_mu)


def plot(losses, xs, log_ws):
    # For plotting
    x_mg, y_mg = jnp.meshgrid(*2 * (jnp.linspace(-8, 8, 200),))
    densities = jax.vmap(lambda x, y: jnp.exp(get_log_mu(jnp.array((x, y)))))(
        x_mg.flatten(), y_mg.flatten()
    ).reshape(x_mg.shape)

    fig = plt.figure(figsize=(12, 3))

    plt.subplot(1, 4, 1)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 4, 2)
    plt.imshow(
        densities,
        origin="lower",
        cmap="Reds",
        extent=(x_mg.min(), x_mg.max(), x_mg.min(), x_mg.max()),
    )
    plt.title("Truth")

    plt.subplot(1, 4, 3)
    plt.hist2d(
        *xs.T,
        bins=jnp.linspace(x_mg.min(), x_mg.max(), 40),  # type: ignore
        weights=jnp.exp(log_ws - log_ws.mean()),
        density=True,
        cmap="Reds",
    )
    plt.gca().set_aspect("equal")
    plt.title("Samples")

    plt.subplot(1, 4, 4)
    try:
        plt.hist(log_ws, bins=50)
    except ValueError as e:
        print(e)
    plt.xlabel(r"$\log w$")
    plt.ylabel("Count")
    plt.title("Weights")

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    key = PRNGKey(9)

    # Set up sampler
    t0 = 0.0
    t1 = 5.0
    n_ts = 100
    dt0 = (t1 - t0) / n_ts
    pis = PathIntegralSampler(X_DIM, t1, dt0)

    # Construct the network
    key, subkey = split(key)
    model = ControlNet(
        subkey,
        X_DIM,
        get_score_mu,
        64,
        3,
        activation=jax.nn.selu,
        weight_init=lecun_init,
        bias_init=zeros_init,
        T=t1,
    )
    lr = 2e-3
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))  # type: ignore
    batch_size = 128

    @eqx.filter_jit
    def train_step(model, key, opt_state):
        loss, grads = eqx.filter_value_and_grad(pis.get_loss)(
            model, key, get_log_mu, batch_size
        )
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    n_steps = 200
    losses = []
    print("training")
    with trange(n_steps) as pbar:
        for _ in pbar:
            key, subkey = split(key)
            loss, model, opt_state = train_step(model, subkey, opt_state)
            if jnp.isnan(loss):
                raise ValueError("hit nan loss")
            losses.append(loss.item())
            pbar.set_postfix(loss=losses[-1])

    losses = jnp.array(losses)

    n_samples = 20000
    print(f"drawing {n_samples} samples")
    key, subkey = split(key)
    xs, log_ws = sample(subkey, pis, model, n_samples)

    print("plotting")
    fig = plot(losses, xs, log_ws)
    fig.savefig("output/gaussian-mixture-2d.png")

    print("done!")
