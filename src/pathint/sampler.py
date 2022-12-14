import sys
from dataclasses import dataclass, field
from typing import Callable, Tuple

import diffrax as dfx
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.random import PRNGKeyArray
from jaxtyping import Array, PyTree  # type: ignore

# Disable host callbacks for errors since it leads to this bug:
# https://github.com/patrick-kidger/diffrax/pull/104
for module_name, module in sys.modules.items():
    if module_name.startswith("diffrax"):
        if hasattr(module, "branched_error_if"):
            module.branched_error_if = lambda *a, **kw: None  # type: ignore


@dataclass
class PathIntegralSampler:
    """
    Class defining loss and sampling functions for the path integral sampler.

    This approach consists of a training objective and sampling procedure for optimal
    control of the stochastic process

    .. math:: \\mathrm{d}\\mathbf{x}_t = \\mathbf{u}_t \\mathrm{d}t + \\mathrm{d}\\mathbf{w}_t ,

    where :math:`\\mathbf{w}_t` is a Wiener process. A network trained to find the
    control policy :math:`\\mathbf{u}_t(t, \\mathbf{x})` such that the loss function
    is minimized causes the above process to yield samples at time :math:`T` with
    the prespecified distribution :math:`\\mu(\\cdot)`. (Distributions and quantities
    at time :math:`t=T` are often referred to as "terminal".) The procedure also
    yields importance sampling weights :math:`w`.

    Notes:
        As explained in the paper, the control policy network is trained by constructing
        an SDE augmented by the trajectory's cost. This implementation uses a similar
        trick to simultaneously sample and compute importance sampling weights using
        any SDE solver.

    """

    get_log_mu: Callable[[Array], Array]
    """:math:`\\log \\mu(x)`, the log of the (unnormalized) terminal density to
    be sampled from.
    """
    x_size: int
    """size of :math:`x` vector.
    """
    t1: float
    """duration of diffusion.
    """
    dt0: float
    """initial timestep size for solver.
    """
    solver: dfx.AbstractSolver = dfx.Euler()
    """SDE solver.
    """
    brownian_motion_tol: float = 1e-3
    """tolerance for `dfx.VirtualBrownianTree`.
    """
    y0: Array = field(init=False)
    """point at which diffusion begins (the origin).
    """

    def __post_init__(self):
        self.y0 = jnp.zeros(self.x_size + 1)

    def get_log_mu_0(self, x: Array) -> Array:
        """
        Gets log probability for the terminal distribution of the uncontrolled process.
        """
        return dist.Normal(scale=jnp.sqrt(self.t1)).log_prob(x).sum()

    def get_drift(
        self, t: Array, x: Array, model: Callable[[Array, Array], Array]
    ) -> Array:
        """
        Gets the drift coefficient for augmented SDE.

        Args:
            t: time.
            x: state variable, with `x[:-1]` corresponding to :math:`x_t` and `x[-1]`
                corresponding to the trajectory's cost (:math:`y_t` in the paper).
            model: control policy network taking :math:`t` and :math:`x_t` as arguments.
        """
        u = model(t, x[:-1])
        return jnp.append(u, 0.5 * (u**2).sum())

    def get_diffusion_train(self, t: Array, x: Array, _) -> Array:
        """
        Gets the diffusion coefficient for the training SDE.

        Args:
            t: time.
            x: state variable, with `x[:-1]` corresponding to :math:`x_t` and `x[-1]`
                corresponding to the trajectory's cost (:math:`y_t` in the paper).
            _: unused argument required by diffrax.
        """
        return jnp.append(jnp.ones(self.x_size), jnp.zeros(1))

    def get_loss(self, model: PyTree, key: PRNGKeyArray):
        """
        Gets loss for a single trajectory.

        Args:
            model: control policy network taking `t` and `x` as arguments.
            key: PRNG key for the trajectory.

        Returns:
            cost: approximation to :math:`\\int_{t_0}^{t_1} \\mathrm{d}t \\frac{1}{2} \\mathbf{u}_t(t, \\mathbf{x}_t ; \\theta) + \\Psi(\\mathbf{x}_T)`,
                where the second term is the terminal cost specified by the training
                procedure.
        """
        brownian_motion = dfx.VirtualBrownianTree(
            0.0, self.t1, self.brownian_motion_tol, (self.x_size + 1,), key
        )
        terms = dfx.MultiTerm(
            dfx.ODETerm(self.get_drift),
            dfx.WeaklyDiagonalControlTerm(self.get_diffusion_train, brownian_motion),
        )
        return self._sample_x_cost(terms, model)[1]

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
        return jnp.append(jnp.eye(self.x_size), u[None, :], axis=0)

    def _sample_x_cost(self, terms: dfx.MultiTerm, model: PyTree) -> Tuple[Array, Array]:
        """
        Helper to get sample and its cost.
        """
        # TODO: custom control term! f is the identity stacked on u.
        y1 = dfx.diffeqsolve(
            terms,
            self.solver,
            0.0,
            self.t1,
            self.dt0,
            self.y0,
            args=model,
            saveat=dfx.SaveAt(t1=True),
        ).ys[-1]
        # Split up augmented state
        x_T = y1[:-1]
        y_T = y1[-1]
        # Add terminal cost
        Psi_T = self.get_log_mu_0(x_T) - self.get_log_mu(x_T)
        cost = y_T + Psi_T
        return x_T, cost

    def get_sample(self, model: PyTree, key: PRNGKeyArray) -> Tuple[Array, Array]:
        """
        Generates a sample. To generate multiple samples, `vmap` over `key`.

        Args:
            model: control policy network taking `t` and `x` as arguments.
            key: PRNG key for the trajectory.

        Returns:
            x_T: sample.
            log_w: log of the importance sampling weight.
        """
        # TODO: custom control term! f is the identity stacked on u.
        brownian_motion = dfx.VirtualBrownianTree(
            0.0, self.t1, self.brownian_motion_tol, (self.x_size,), key
        )
        terms = dfx.MultiTerm(
            dfx.ODETerm(self.get_drift),
            dfx.ControlTerm(self.get_diffusion_sampling, brownian_motion),
        )
        x_T, cost = self._sample_x_cost(terms, model)
        log_w = -cost
        return x_T, log_w
