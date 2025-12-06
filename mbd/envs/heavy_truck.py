import jax
from jax import numpy as jnp
from flax import struct
from functools import partial
import matplotlib.pyplot as plt


@struct.dataclass
class State:
    pipeline_state: jnp.ndarray
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


class HeavyTruckFuel:
    """5 km highway eco-driving with a heavy truck.

    The controller modulates throttle/brake to minimize fuel while keeping the total
    travel time under a soft cap. The road has a deterministic pitch profile so the
    solver can anticipate grades. Actions are a single scalar in [-1, 1] where
    positive values are throttle, negative values are brakes, and values near zero
    correspond to coasting.
    """

    def __init__(self):
        # Simulation / horizon setup
        self.dt = 1.0
        self.H = 300
        self.road_length = 5000.0  # meters
        self.target_avg_speed = 19.4  # m/s (~70 km/h) time-efficiency target
        self.target_time = self.road_length / self.target_avg_speed  # ~257 s
        self.finish_time_slack = 5.0  # grace window around target time
        self.time_limit = self.target_time + self.finish_time_slack  # used for obs scaling only
        self.hard_time_limit = 320.0  # after this the rollout terminates early

        # Vehicle parameters (rough heavy-duty tractor values)
        self.mass = 30000.0  # kg
        self.gravity = 9.81
        self.c_rr = 0.005  # rolling resistance coefficient
        self.rho_air = 1.2  # kg / m^3
        self.cd = 0.6
        self.frontal_area = 10.0  # m^2
        self.max_traction_force = 35000.0  # N, limited by powertrain/traction
        self.max_brake_force = 65000.0  # N
        self.engine_efficiency = 0.4  # tank-to-wheel efficiency
        self.fuel_energy_density = 36e6  # J / L (diesel)
        self.idle_fuel_rate = 0.00035  # L / s
        self.extra_throttle_fuel = 0.0008  # enrichment term at high throttle

        # Reward shaping parameters
        self.fuel_penalty_scale = 1.0  # primary objective: minimize fuel (L)
        self.finish_time_penalty_scale = 2.0  # per-second penalty outside slack around target_time
        self.timeout_penalty = 200.0  # penalty when time runs out before finishing
        self.unfinished_penalty_scale = 400.0  # scaled by remaining distance if timeout
        self.speed_violation_scale = 80.0  # penalty weight for min/max speed violations
        self.brake_penalty_scale = 1.5  # discourage brake taps

        # Speed preferences (keeps solutions time-efficient but not reckless)
        self.max_speed = 26.4  # 95 km/h realistic heavy-truck upper bound
        self.min_speed = 60 / 3.6
        self.start_speed = 18.0  # ~65 km/h initial velocity; non-zero for realism
        self.fuel_normalizer = 20.0  # expected fuel burn upper bound for obs scaling
        self.rew_xref = 0.0

        # Pitch profile along the 5 km stretch (degrees -> radians)
        self.pitch_positions = jnp.array(
            [
                0.0,
                500.0,
                1000.0,
                1600.0,
                2200.0,
                2800.0,
                3400.0,
                4000.0,
                4600.0,
                5000.0,
            ],
            dtype=jnp.float32,
        )
        self.pitch_angles = jnp.deg2rad(
            jnp.array(
                # Moderately rolling terrain to balance difficulty and feasibility.
                [0.0, 1.5, 3.0, 2.5, -1.0, -3.0, -0.5, 2.5, 3.0, 0.0],
                dtype=jnp.float32,
            )
        )

    def reset(self, rng: jax.Array) -> State:
        """Resets to the start of the highway section."""
        del rng  # unused, kept for API compatibility
        pipeline_state = jnp.array(
            [0.0, self.start_speed, 0.0, 0.0], dtype=jnp.float32
        )  # [distance (m), speed (m/s), fuel_used (L), time (s)]
        obs = self._get_obs(pipeline_state)
        return State(pipeline_state, obs, 0.0, 0.0)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        """One step of longitudinal dynamics."""
        act = jnp.squeeze(jnp.clip(action, -1.0, 1.0))
        throttle = jnp.clip(act, 0.0, 1.0)
        brake = jnp.clip(-act, 0.0, 1.0)

        s, v, fuel_used, t = state.pipeline_state
        grade = self._grade_at(s)

        drag_force = 0.5 * self.rho_air * self.cd * self.frontal_area * v**2
        rolling_force = self.mass * self.gravity * self.c_rr * jnp.cos(grade)
        grade_force = self.mass * self.gravity * jnp.sin(grade)
        traction = throttle * self.max_traction_force
        braking = brake * self.max_brake_force

        net_force = traction - drag_force - rolling_force - grade_force - braking
        acc = net_force / self.mass
        # Keep speed physical (non-negative), but avoid hard-clipping to max; violations are penalized in reward.
        v_new = jnp.maximum(v + acc * self.dt, 0.0)
        # Use simple trapezoidal integration for distance.
        s_new = s + jnp.maximum((v + v_new) * 0.5, 0.0) * self.dt
        t_new = t + self.dt

        fuel_rate = self._fuel_rate(throttle, v_new, traction)
        fuel_new = fuel_used + fuel_rate * self.dt

        pipeline_state = jnp.array([s_new, v_new, fuel_new, t_new])
        obs = self._get_obs(pipeline_state)
        reward = self._get_reward(s, s_new, fuel_rate, t_new, v_new, throttle, brake)
        done = jnp.where(
            (s_new >= self.road_length) | (t_new >= self.hard_time_limit),
            1.0,
            0.0,
        )

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    @partial(jax.jit, static_argnums=(0,))
    def _grade_at(self, s: jax.Array) -> jax.Array:
        s_clamped = jnp.clip(s, 0.0, self.road_length)
        return jnp.interp(s_clamped, self.pitch_positions, self.pitch_angles)

    @partial(jax.jit, static_argnums=(0,))
    def _fuel_rate(
        self, throttle: jax.Array, speed: jax.Array, traction_force: jax.Array
    ) -> jax.Array:
        power = traction_force * jnp.maximum(speed, 1.0)  # prevent zero-division issues
        useful_power = jnp.maximum(power, 0.0)
        fuel_from_power = useful_power / (self.engine_efficiency * self.fuel_energy_density)
        return self.idle_fuel_rate + throttle * self.extra_throttle_fuel + fuel_from_power

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(
        self,
        s_prev: jax.Array,
        s_new: jax.Array,
        fuel_rate: jax.Array,
        t_new: jax.Array,
        v_new: jax.Array,
        throttle: jax.Array,
        brake: jax.Array,
    ) -> jax.Array:
        fuel_cost = fuel_rate * self.dt * self.fuel_penalty_scale
        # Only accrue costs while we are still before the goal and time limit.
        active = (s_prev < self.road_length) & (t_new < self.hard_time_limit)
        base_reward = jnp.where(active, -fuel_cost, 0.0)

        # Trigger terminal incentives only on the first crossing of the finish line.
        reached_now = (s_prev < self.road_length) & (s_new >= self.road_length)
        remaining = jnp.maximum(self.road_length - s_new, 0.0)
        time_over_target = jnp.maximum(t_new - (self.target_time + self.finish_time_slack), 0.0)
        finish_time_penalty = jnp.where(
            reached_now, self.finish_time_penalty_scale * time_over_target, 0.0
        )

        # If time runs out before finishing, penalize the distance left.
        timed_out = (t_new >= self.hard_time_limit) & (~reached_now) & (s_prev < self.road_length)
        timeout_penalty = jnp.where(
            timed_out,
            self.timeout_penalty + self.unfinished_penalty_scale * (remaining / self.road_length),
            0.0,
        )

        speed_over = jnp.maximum(v_new - self.max_speed, 0.0)
        speed_under = jnp.maximum(self.min_speed - v_new, 0.0)
        speed_violation_penalty = jnp.where(
            active, self.speed_violation_scale * (speed_over**2 + speed_under**2), 0.0
        )
        brake_penalty = jnp.where(active, self.brake_penalty_scale * (brake**2), 0.0)

        return (
            base_reward
            - finish_time_penalty
            - timeout_penalty
            - speed_violation_penalty
            - brake_penalty
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, pipeline_state: jax.Array) -> jax.Array:
        s, v, fuel_used, t = pipeline_state
        grade_now = self._grade_at(s)
        grade_ahead = self._grade_at(jnp.minimum(s + 200.0, self.road_length))
        remaining = jnp.maximum(self.road_length - s, 0.0)
        return jnp.array(
            [
                s / self.road_length,
                v / self.max_speed,
                grade_now,
                grade_ahead,
                fuel_used / self.fuel_normalizer,
                t / self.time_limit,
            ]
        )

    @property
    def action_size(self):
        return 1

    @property
    def observation_size(self):
        return 6

    def warm_start_actions(self, H: int) -> jax.Array:
        """Heuristic open-loop to hold target speed; used as diffusion prior."""
        # Assume evenly spaced distance progression toward goal.
        distances = jnp.linspace(0.0, self.road_length, H)
        v_ref = self.target_avg_speed
        grade = self._grade_at(distances)
        drag_force = 0.5 * self.rho_air * self.cd * self.frontal_area * v_ref**2
        rolling_force = self.mass * self.gravity * self.c_rr * jnp.cos(grade)
        grade_force = self.mass * self.gravity * jnp.sin(grade)
        required_force = drag_force + rolling_force + grade_force
        throttle = jnp.clip(required_force / self.max_traction_force, 0.0, 1.0)
        return throttle[:, None]

    def eval_xref_logpd(self, xs: jax.Array) -> jax.Array:
        """Likelihood of following an eco-cruise speed profile (for demo mixing)."""
        raise NotImplementedError("eval_xref_logpd is not implemented for heavy_truck")

    def render(
        self,
        axes,
        xs: jnp.ndarray,
        us: jnp.ndarray | None = None,
        warm_start: jnp.ndarray | None = None,
    ):
        """Plot rollout with separated panels to avoid overlap."""
        s = jnp.array(xs[:, 0])
        v = jnp.array(xs[:, 1]) * 3.6  # km/h
        fuel = jnp.array(xs[:, 2])
        t = jnp.array(xs[:, 3])
        grade = jnp.rad2deg(self._grade_at(s))
        dist_km = s / 1000.0
        fuel_rate = jnp.concatenate([jnp.diff(fuel), jnp.array([fuel[-1] - fuel[-2]])]) / self.dt
        avg_speed_trace = s / jnp.maximum(t, 1e-3) * 3.6  # km/h
        finished_mask = s >= self.road_length
        finish_idx = jnp.where(finished_mask.any(), jnp.argmax(finished_mask), s.shape[0] - 1)

        if len(axes) < 4:
            raise ValueError("Expected at least four subplots for heavy-truck rendering.")

        ax_speed = axes[0]
        speed_line, = ax_speed.plot(dist_km, v, label="speed (km/h)", color="tab:blue")
        ax_speed.plot(dist_km, avg_speed_trace, color="tab:purple", linestyle="--", label="avg speed (km/h)")
        ax_speed.axhline(self.target_avg_speed * 3.6, color="tab:purple", linestyle=":", alpha=0.8, label="target avg")
        ax_speed.axhline(self.max_speed * 3.6, color="tab:red", linestyle=":", alpha=0.7, label="max limit")
        ax_speed.axhline(self.min_speed * 3.6, color="tab:red", linestyle=":", alpha=0.7, label="min limit")
        ax_speed.set_ylabel("Speed (km/h)")
        ax_speed.grid(True, linestyle="--", alpha=0.5)
        ax_speed.set_xlim(0.0, self.road_length / 1000.0)
        ax_speed.set_title("Heavy truck eco-driving rollout")
        ax_speed.legend(loc="upper left")

        ax_grade = axes[1]
        ax_grade.plot(dist_km, grade, "r--", label="pitch (deg)")
        ax_grade.axhline(0.0, color="k", linestyle=":", alpha=0.5)
        ax_grade.set_ylabel("Pitch (deg)")
        ax_grade.set_xlim(0.0, self.road_length / 1000.0)
        ax_grade.grid(True, linestyle="--", alpha=0.5)
        ax_grade.legend(loc="upper left")

        ax_ctrl = axes[2]
        if us is not None:
            controls = jnp.squeeze(us)
            throttle = jnp.clip(controls, 0.0, 1.0)
            brake = jnp.clip(-controls, 0.0, 1.0)
            # Actions have one fewer element than states.
            dist_ctrl = dist_km[: throttle.shape[0]]
            ax_ctrl.fill_between(
                dist_ctrl, 0, throttle, color="tab:green", alpha=0.25, label="throttle"
            )
            ax_ctrl.fill_between(
                dist_ctrl, 0, brake, color="tab:red", alpha=0.25, label="brake"
            )
            ax_ctrl.plot(dist_ctrl, throttle, color="tab:green", alpha=0.9)
            ax_ctrl.plot(dist_ctrl, brake, color="tab:red", alpha=0.9)
        if warm_start is not None:
            ws = jnp.squeeze(warm_start)
            ws_dist = dist_km[: ws.shape[0]]
            ax_ctrl.plot(ws_dist, ws, color="tab:blue", linestyle="--", label="warm start action")
        ax_ctrl.set_ylabel("Control (throttle/brake)")
        ax_ctrl.set_xlim(0.0, self.road_length / 1000.0)
        ax_ctrl.set_xlabel("Distance (km)")
        ax_ctrl.grid(True, linestyle="--", alpha=0.5)
        ax_ctrl.legend(loc="upper left")

        ax_fuel = axes[3]
        ax_fuel.plot(dist_km, fuel, color="tab:blue", label="fuel used (L)")
        ax_fuel_rate = ax_fuel.twinx()
        ax_fuel_rate.plot(dist_km, fuel_rate * 3600, color="tab:orange", alpha=0.8, label="fuel rate (L/hr)")
        ax_fuel.set_ylabel("Fuel used (L)")
        ax_fuel_rate.set_ylabel("Fuel rate (L/hr)", color="tab:orange")
        ax_fuel_rate.tick_params(axis="y", colors="tab:orange")
        ax_fuel.set_xlabel("Distance (km)")
        ax_fuel.grid(True, linestyle="--", alpha=0.5)
        ax_fuel.set_xlim(0.0, self.road_length / 1000.0)
        ax_fuel.legend(loc="upper left")
        ax_fuel_rate.legend(loc="upper right")

        # Annotate summary stats.
        end_time = float(t[int(finish_idx)])
        total_fuel = float(fuel[int(finish_idx)])
        avg_speed_final = float(s[int(finish_idx)] / max(t[int(finish_idx)], 1e-3) * 3.6)
        ax_fuel.text(
            0.01,
            0.92,
            f"Total fuel: {total_fuel:.2f} L\nTime: {end_time:.0f} s\nAvg speed: {avg_speed_final:.1f} km/h",
            transform=ax_fuel.transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
