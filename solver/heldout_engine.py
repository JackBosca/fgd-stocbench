"""Compiled snapshot evolution with one RNG stream per simulation/member.

Reuses the upstream spatial operators and Euler updates. Noise draws have a
fixed single-field shape so job sharding and batching do not redefine streams.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from solver.configs import IncnsConfig
from solver.incns_stoc import StocIncNSSolver


class HeldoutEngine:
    def __init__(self, config, variant):
        self.variant = variant
        self.config = config
        self.solver = StocIncNSSolver(IncnsConfig(**config["solver"]))
        self.solver._setup()
        s = self.solver

        def noise(key):
            key, sub = jax.random.split(key)
            coefficients = jax.random.normal(sub, (s.forcing_basis.shape[0],))
            return key, jnp.tensordot(coefficients, s.forcing_basis, axes=1)

        def drift(wh):
            wh = wh.at[..., 0, 0].set(0.0)
            return wh + s.dt * (s.linear_hat * wh + s._nonlinear_head(wh))

        def stochastic_interval(wh, key):
            def body(_, state):
                w, k = state
                k, forcing = noise(k)
                return drift(w) + s.cfg.epsilon * jnp.sqrt(s.dt) * forcing, k

            return jax.lax.fori_loop(0, s.steps, body, (wh, key))

        def deterministic_interval(wh, forcing):
            scale = s.cfg.epsilon * s.dt / jnp.sqrt(s.cfg.sample_dt)
            return jax.lax.fori_loop(0, s.steps, lambda _, w: drift(w) + scale * forcing, wh)

        self.stochastic = jax.jit(jax.vmap(stochastic_interval))
        self.deterministic = jax.jit(jax.vmap(deterministic_interval))
        self.forcing = jax.jit(jax.vmap(noise))
        self.fft = jax.jit(lambda w: jnp.fft.fft2(w))
        self.ifft = jax.jit(lambda w: jnp.fft.ifft2(w).real)

    def advance(self, wh, keys):
        if self.variant == "stoc":
            return self.stochastic(wh, keys), None
        keys, forcing = self.forcing(keys)
        return (self.deterministic(wh, forcing), keys), forcing

    def warm(self, keys, progress=None):
        n = self.config["solver"]["n"]
        wh = jnp.zeros((len(keys), n, n), dtype=jnp.complex64)
        keys = jnp.asarray(keys)
        for frame in range(self.config["warmup_frames"]):
            (wh, keys), _ = self.advance(wh, keys)
            if progress and (frame + 1) % 10 == 0:
                progress(f"warmup={frame + 1}/{self.config['warmup_frames']}")
        return wh

    def normalized(self, wh, grid=None):
        c = self.config["solver"]
        w = (self.ifft(wh) - c["mean"]) / c["std"]
        return np.asarray(self.solver._downsample(w, grid or self.config["grid"]), dtype=np.float32)

    def trajectories(self, warm_keys, trajectory_keys, progress=None):
        wh = self.warm(warm_keys, progress)
        keys = jnp.asarray(trajectory_keys)
        batch, frames, grid = len(keys), self.config["trajectory_frames"], self.config["grid"]
        out = np.empty((batch, frames, 1 if self.variant == "stoc" else 2, grid, grid), dtype=np.float32)
        for frame in range(frames):
            if self.variant == "stoc":
                # Upstream stochastic files first record after one trajectory interval.
                (wh, keys), _ = self.advance(wh, keys)
                out[:, frame, 0] = self.normalized(wh)
            else:
                # Upstream deterministic files record state and the FOLLOWING forcing.
                keys, forcing = self.forcing(keys)
                out[:, frame, 0] = self.normalized(wh)
                out[:, frame, 1] = np.asarray(self.solver._downsample(self.ifft(forcing), grid))
                wh = self.deterministic(wh, forcing)
            if progress and (frame + 1) % 10 == 0:
                progress(f"frame={frame + 1}/{frames}")
        return out

    def condition(self, warm_key, state_key):
        wh = self.warm(np.asarray([warm_key]))
        (wh, _), _ = self.advance(wh, np.asarray([state_key]))
        # Preserve the high-resolution normalized state used for every branch.
        return self.normalized(wh, self.config["solver"]["n"])

    def futures(self, high_resolution_init, keys):
        c = self.config["solver"]
        physical = high_resolution_init[0] * c["std"] + c["mean"]
        wh = self.fft(jnp.broadcast_to(physical, (len(keys), *physical.shape)))
        wh, _ = self.stochastic(wh, jnp.asarray(keys))
        return self.normalized(wh)
