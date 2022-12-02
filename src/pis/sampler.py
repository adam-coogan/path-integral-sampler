import sys
from dataclasses import dataclass, field
from typing import Callable, Tuple

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.random import PRNGKeyArray, split
from jaxtyping import Array, PyTree  # type: ignore

# Disable host callbacks for errors since it leads to this bug:
# https://github.com/patrick-kidger/diffrax/pull/104
for module_name, module in sys.modules.items():
    if module_name.startswith("diffrax"):
        if hasattr(module, "branched_error_if"):
            module.branched_error_if = lambda *a, **kw: None  # type: ignore

# TODO:
# - Refactor so key is always first
# - Factor training out of this class!


@dataclass
class PathIntegralSampler:
    """
    Class defining loss and sampling functions for the path integral sampler (PIS).

    PIS consists of a training objective and sampling procedure for optimal control
    of the stochastic process

    .. math:: \mathrm{d}\mathbf{x}_t = \mathbf{u}_t \mathrm{d}t + \mathrm{d}\mathbf{w}_t ,

    where :math:`\mathbf{w}_t` is a Wiener process. A network trained to find the
    control policy :math:`\mathbf{u}_t(t, \mathbf{x})` such that the PIS loss function
    is minimized causes the above process to yield samples at time :math:`T` with
    the prespecified distribution :math:`\mu(\cdot)`. (Distributions and quantities
    at time :math:`t=T` are often referred to as "terminal".) The procedure also
    yields importance sampling weights :math:`w`.

    Notes:
        As explained in the paper, the control policy network is trained by constructing
        an SDE augmented by the trajectory's cost. In this implementation, I've
        used a similar trick to simultaneously sample and compute importance sampling
        weights using any SDE solver.

    """

    x_dim: int
    t1: float
    dt0: float
    solver: dfx.AbstractSolver = dfx.Euler()
    t0: float = 0.0
    brownian_motion_tol: float = 1e-3
    y0: Array = field(init=False)

    def __post_init__(self):
        self.y0 = jnp.zeros(self.x_dim + 1)

    def get_log_mu_0(self, x: Array) -> Array:
        """
        Gets log probability for the terminal distribution of the uncontrolled process.
        """
        return dist.Normal(scale=jnp.sqrt(jnp.abs(self.t1 - self.t0))).log_prob(x).sum()

    def get_drift_train(
        self, t: Array, x: Array, model: Callable[[Array, Array], Array]
    ) -> Array:
        """
        Gets the drift coefficient for the training SDE.

        Args:
            t: time.
            x: position.
            model: control policy network taking `t` and `x` as arguments.
        """
        u = model(t, x[:-1])
        return jnp.append(u, 0.5 * (u**2).sum())

    def get_diffusion_train(self, t: Array, x: Array, _) -> Array:
        """
        Gets the diffusion coefficient for the training SDE.

        Args:
            t: time.
            x: position.
            model: control policy network taking `t` and `x` as arguments.
        """
        return jnp.append(jnp.ones(self.x_dim), jnp.zeros(1))

    def get_x_T_cost_trajectory(
        self, model: PyTree, key: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        """
        Gets the terminal sample and cost along the trajectory for the given model.

        Args:
            model: control policy network taking `t` and `x` as arguments.
            key: PRNG key for the trajectory.

        Returns:
            x_T: terminal sample.
            cost_trajectory: approximation to :math:`\int_{t_0}^{t_1} \mathrm{d}t \frac{1}{2} \mathbf{u}_t(t, \mathbf{x}_t ; \theta)`.
        """
        # TODO: custom control term! Don't need last component of the Brownian motion.
        brownian_motion = dfx.VirtualBrownianTree(
            self.t0, self.t1, self.brownian_motion_tol, (self.x_dim + 1,), key
        )
        terms = dfx.MultiTerm(
            dfx.ODETerm(self.get_drift_train),
            dfx.WeaklyDiagonalControlTerm(self.get_diffusion_train, brownian_motion),
        )
        y1 = dfx.diffeqsolve(
            terms,
            self.solver,
            self.t0,
            self.t1,
            self.dt0,
            self.y0,
            args=model,
            saveat=dfx.SaveAt(t1=True),
        ).ys[-1]
        x_T = y1[:-1]
        cost_trajectory = y1[-1]
        return x_T, cost_trajectory

    def _get_loss_1(
        self,
        model: PyTree,
        key: PRNGKeyArray,
        get_log_mu: Callable[[Array], Array],
    ):
        """
        Gets loss for a single trajectory.

        Args:
            model: control policy network taking `t` and `x` as arguments.
            key: PRNG key for the trajectory.
            get_log_mu: function returning (potentially unnormalized) terminal log
                probability.

        Returns:
            cost: approximation to :math:`\int_{t_0}^{t_1} \mathrm{d}t \frac{1}{2} \mathbf{u}_t(t, \mathbf{x}_t ; \theta) + \Psi(\mathbf{x}_T)`,
                where the second term is the terminal cost specified by the PIS
                training procedure.
        """
        x_T, cost_trajectory = self.get_x_T_cost_trajectory(model, key)
        cost_terminal = self.get_log_mu_0(x_T) - get_log_mu(x_T)
        cost = cost_trajectory + cost_terminal
        return cost

    def get_loss(
        self,
        model: PyTree,
        key: PRNGKeyArray,
        get_log_mu: Callable[[Array], Array],
        batch_size: int,
    ):
        """
        Gets loss for a batch of trajectories.

        Args:
            model: control policy network taking `t` and `x` as arguments.
            key: PRNG key for the trajectory.
            get_log_mu: function returning (potentially unnormalized) terminal log
                probability.
            batch_size: number of trajectories to sample in the batch.
        """
        keys = jnp.stack(split(key, batch_size))  # type: ignore
        in_axes = (None, 0, None)
        return jax.vmap(self._get_loss_1, in_axes)(model, keys, get_log_mu).sum()

    def get_drift_sampling(
        self, t: Array, x: Array, model: Callable[[Array, Array], Array]
    ) -> Array:
        """
        Gets the drift coefficient for sampling.

        Args:
            t: time.
            x: position.
            model: control policy network taking `t` and `x` as arguments.
        """
        u = model(t, x[:-1])
        return jnp.append(u, 0.5 * (u**2).sum())

    def get_diffusion_sampling(
        self, t: Array, x: Array, model: Callable[[Array, Array], Array]
    ) -> Array:
        """
        Gets the diffusion coefficient for sampling.

        Args:
            t: time.
            x: position.
            model: control policy network taking `t` and `x` as arguments.
        """
        u = model(t, x[:-1])
        return jnp.append(jnp.eye(self.x_dim), u[None, :], axis=0)

    def get_sample(
        self, model: PyTree, key, get_log_mu: Callable[[Array], Array]
    ) -> Tuple[Array, Array]:
        """
        Generates a sample. To generate multiple samples, `vmap` over `key`.

        Args:
            model: control policy network taking `t` and `x` as arguments.
            key: PRNG key for the trajectory.
            get_log_mu: function returning (potentially unnormalized) terminal log
                probability.

        Returns:
            x_T: sample.
            log_w: log of the importance sampling weight.
        """
        # TODO: custom control term! f is the identity stacked on u.
        brownian_motion = dfx.VirtualBrownianTree(
            self.t0, self.t1, self.brownian_motion_tol, (self.x_dim,), key
        )
        terms = dfx.MultiTerm(
            dfx.ODETerm(self.get_drift_sampling),
            dfx.ControlTerm(self.get_diffusion_sampling, brownian_motion),
        )
        y1 = dfx.diffeqsolve(
            terms,
            self.solver,
            self.t0,
            self.t1,
            self.dt0,
            self.y0,
            args=model,
            saveat=dfx.SaveAt(t1=True),
        ).ys[-1]
        x_T = y1[:-1]
        cost_trajectory = y1[-1]
        cost_terminal = self.get_log_mu_0(x_T) - get_log_mu(x_T)
        log_w = -(cost_trajectory + cost_terminal)
        return x_T, log_w
