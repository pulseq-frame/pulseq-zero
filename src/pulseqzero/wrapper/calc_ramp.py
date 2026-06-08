"""Differentiable port of `pypulseq.calc_ramp` (v1.5.0.post1).

This port was written by Claude Code and might not be up to the same quality standards as the rest of pulseq-zero.

`calc_ramp` joins two points in 3D k-space in minimal time while respecting the
gradient and slew limits, given the gradient `G0` before the start point and
`G_end` after the end point. The number of connecting points is found by a
discrete search (try 0 points, then 1, then 2, ... until all constraints can be
met) and is inherently non-differentiable, but the connecting point *positions*
are computed with torch ops, so gradients flow from `k0` / `k_end` into the
returned trajectory - the same boundary as the other adapter functions.

Simplifications / fixes vs. the upstream pypulseq implementation:

- Only scalar (Euclidean) gradient/slew limits are supported. PyPulseq's
  per-axis mode (passing length-3 `max_grad` / `max_slew`) had inconsistent
  `grad_raster` powers between the left/right recursion and printed
  ``'Unknown error'`` instead of raising; it is dropped here. Pass scalar
  limits, or fall back to `seq.to_pypulseq()` for the per-axis case.
- The scalar path is actually usable for more than one connecting point.
  PyPulseq's scalar recursion forwarded its arguments positionally in the
  wrong order (a gradient vector landed where `use_points` was expected, which
  crashes on the next `use_points == 0` test). The correct argument order -
  visible in PyPulseq's keyword-based per-axis recursion - is used here.
- The feasibility check tests the per-sample gradient/slew vector magnitude
  (sum over the spatial axes). PyPulseq summed over the time axis instead,
  disagreeing with its own placement logic.
- The two-sphere intersection uses a correct vector projection (PyPulseq used
  an element-wise product where a dot product was intended).
"""

from typing import Optional, cast

import torch

from pypulseq import Opts

from ..events import Array, Scalar


def calc_ramp(
    k0: Array,
    k_end: Array,
    max_grad: Optional[Scalar] = None,
    max_points: int = 500,
    max_slew: Optional[Scalar] = None,
    system: Optional[Opts] = None,
    oversampling: bool = False,
) -> tuple[Array, bool]:
    """Join `k0` and `k_end` in k-space in minimal time within grad/slew limits.

    Parameters
    ----------
    k0 : array_like, shape [3, 2]
        The two preceding k-space points. The starting gradient is derived from
        them as `(k0[:, 1] - k0[:, 0]) / grad_raster`.
    k_end : array_like, shape [3, 2]
        The two following k-space points. The target gradient is derived as
        `(k_end[:, 1] - k_end[:, 0]) / grad_raster`.
    max_grad : float, optional
        Maximum gradient strength (Euclidean). Defaults to `system.max_grad`.
    max_points : int, default=500
        Maximum number of connecting points to try.
    max_slew : float, optional
        Maximum slew rate (Euclidean). Defaults to `system.max_slew`.
    system : Opts, optional
        System limits. Defaults to `Opts.default`.
    oversampling : bool, default=False
        If True, the gradient is oversampled by a factor of two (half raster).

    Returns
    -------
    k_out : torch.Tensor, shape [3, n]
        The connecting k-space points (autograd-connected to `k0` / `k_end`).
    success : bool
        Whether `k0` and `k_end` were successfully joined.
    """
    if system is None:
        system = Opts.default

    if max_grad is None:
        max_grad = cast(float, system.max_grad)
    if max_slew is None:
        max_slew = cast(float, system.max_slew)

    grad_raster = system.grad_raster_time
    if oversampling:
        grad_raster = 0.5 * grad_raster

    k0 = torch.as_tensor(k0)
    k_end = torch.as_tensor(k_end)

    # Multiplicative slack to keep points constructed exactly on a limit from
    # being rejected by float noise (the analytic placements land on the limit).
    tol = 1.0 + 1e-9

    def sq_norm(v: torch.Tensor) -> torch.Tensor:
        return (v**2).sum()

    def inside_limits(grad: torch.Tensor, slew: torch.Tensor) -> bool:
        # grad: [3 axes, n], slew: [3 axes, n - 1]. Per-sample vector magnitude.
        grad_ok = bool(torch.all((grad**2).sum(0) <= (max_grad * tol) ** 2))
        slew_ok = bool(torch.all((slew**2).sum(0) <= (max_slew * tol) ** 2))
        return grad_ok and slew_ok

    def place(p: torch.Tensor, dk: torch.Tensor, dkprol: torch.Tensor) -> torch.Tensor:
        """Place one point as far along `dk` from `p` as the limits allow.

        `dkprol` is the prolongation of the previous gradient (`G_prev *
        grad_raster`). The grad limit confines the point to a sphere of radius
        `a` around `p`; the slew limit to a sphere of radius `b` around
        `p + dkprol`. We step the full budget along the connecting direction and
        fall back to the intersection of both spheres when neither pure step
        fits.
        """
        a = max_grad * grad_raster  # max |point - p|
        b = max_slew * grad_raster**2  # max |point - p - dkprol|

        # Slew-limited: spend the whole slew budget along the connection.
        dkconn = dk - dkprol
        ksl = p + dkprol + dkconn / torch.linalg.norm(dkconn) * b
        if bool(sq_norm(ksl - p) <= (a * tol) ** 2):
            return ksl

        # Grad-limited: spend the whole grad budget along the ideal direction.
        kgl = p + dk / torch.linalg.norm(dk) * a
        if bool(sq_norm(kgl - p - dkprol) <= (b * tol) ** 2):
            return kgl

        # Both limits active: intersection circle of the two spheres, taking the
        # point on the `kgl` side.
        c = torch.linalg.norm(dkprol)
        u = dkprol / c
        c1 = (a**2 - b**2 + c**2) / (2 * c)
        h = torch.sqrt(torch.clamp(a**2 - c1**2, min=0.0))
        perp = (kgl - p) - torch.dot(kgl - p, u) * u
        return p + c1 * u + h * perp / (torch.linalg.norm(perp) + 1e-30)

    def join_left(k_start, k_stop, g_in, g_out, n):
        if n == 0:
            mid = (k_stop - k_start) / grad_raster
            grad = torch.stack((g_in, mid, g_out), dim=1)
            slew = (grad[:, 1:] - grad[:, :-1]) / grad_raster
            return inside_limits(grad, slew), k_start.new_zeros((3, 0))

        dk = (k_stop - k_start) / (n + 1)
        g_opt = dk / grad_raster
        s_opt = (g_opt - g_in) / grad_raster
        if bool(sq_norm(g_opt) <= (max_grad * tol) ** 2) and bool(
            sq_norm(s_opt) <= (max_slew * tol) ** 2
        ):
            k_left = k_start + dk
        else:
            k_left = place(k_start, dk, g_in * grad_raster)

        ok, rest = join_right(k_left, k_stop, (k_left - k_start) / grad_raster, g_out, n - 1)
        return ok, torch.cat((k_left[:, None], rest), dim=1)

    def join_right(k_start, k_stop, g_in, g_out, n):
        if n == 0:
            mid = (k_stop - k_start) / grad_raster
            grad = torch.stack((g_in, mid, g_out), dim=1)
            slew = (grad[:, 1:] - grad[:, :-1]) / grad_raster
            return inside_limits(grad, slew), k_start.new_zeros((3, 0))

        dk = (k_start - k_stop) / (n + 1)
        g_opt = -dk / grad_raster
        s_opt = (g_out - g_opt) / grad_raster
        if bool(sq_norm(g_opt) <= (max_grad * tol) ** 2) and bool(
            sq_norm(s_opt) <= (max_slew * tol) ** 2
        ):
            k_right = k_stop + dk
        else:
            k_right = place(k_stop, dk, -g_out * grad_raster)

        ok, rest = join_left(k_start, k_right, g_in, (k_stop - k_right) / grad_raster, n - 1)
        return ok, torch.cat((rest, k_right[:, None]), dim=1)

    # Gradients at the connection endpoints.
    g0 = (k0[:, 1] - k0[:, 0]) / grad_raster
    g_end = (k_end[:, 1] - k_end[:, 0]) / grad_raster
    start = k0[:, 1]
    stop = k_end[:, 0]

    # Endpoints already exceed the gradient limit -> impossible to connect.
    if bool(torch.linalg.norm(g0) > max_grad * tol) or bool(
        torch.linalg.norm(g_end) > max_grad * tol
    ):
        return start.new_zeros((3, 0)), False

    k_out = start.new_zeros((3, 0))
    success = False
    for n in range(max_points + 1):
        success, k_out = join_left(start, stop, g0, g_end, n)
        if success:
            break

    return k_out, success
