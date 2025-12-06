import functools
import os
import jax
from jax import numpy as jnp
from jax import config
from dataclasses import dataclass
import tyro
from tqdm import tqdm
from matplotlib import pyplot as plt

import mbd

# NOTE: enable this if you want higher precision
# config.update("jax_enable_x64", True)


## load config
@dataclass
class Args:
    # exp
    seed: int = 0  # PRNG seed
    disable_recommended_params: bool = (
        False  # keep True to force manual hyperparameters
    )
    not_render: bool = False  # skip writing HTML/fig output
    # env
    env_name: str = (
        "ant"  # "humanoidstandup", "ant", "halfcheetah", "hopper", "walker2d", "car2d"
    )
    # diffusion
    Nsample: int = 2048  # trajectories sampled per reverse step
    Hsample: int = 50  # planning horizon (timesteps)
    Ndiffuse: int = 100  # reverse diffusion steps (noise levels)
    temp_sample: float = 0.1  # softmax temperature on returns
    beta0: float = 1e-4  # initial beta for noise schedule
    betaT: float = 1e-2  # final beta for noise schedule
    enable_demo: bool = False  # when True, mix demo likelihoods into weights


def run_diffusion(args: Args):
    """Monte-Carlo score-ascent diffusion over action sequences via model rollouts.

    Step-by-step:
      1) Build a noise schedule (betas -> alphas_bar, sigmas).
      2) Initialize a noisy action trajectory (all zeros here).
      3) For i = Ndiffuse-1 down to 1:
         a) Sample many trajectories from q_i (inject noise at level i).
         b) Roll out each trajectory in the dynamics to get rewards.
         c) Normalize rewards -> softmax weights (optionally fused with demos).
         d) Reward-weighted mean of trajectories is the Monte-Carlo score proxy.
         e) Use that score proxy to take a reverse/noise-reduction step.
      4) Return the final low-noise trajectory and its mean reward.
    """
    rng = jax.random.PRNGKey(seed=args.seed)

    ## setup env

    # Recommended hyperparameters per environment (empirically tuned for stability).
    temp_recommend = {
        "ant": 0.1,
        "halfcheetah": 0.4,
        "hopper": 0.1,
        "humanoidstandup": 0.1,
        "humanoidrun": 0.1,
        "walker2d": 0.1,
        "pushT": 0.2,
        "heavytruck": 0.2,
    }
    Ndiffuse_recommend = {
        "pushT": 200,
        "humanoidrun": 300,
        "heavytruck": 1000,
    }
    Nsample_recommend = {
        "humanoidrun": 8192,
        "heavytruck": 8192,
    }
    Hsample_recommend = {
        "pushT": 40,
        "heavytruck": 300,
    }
    beta_recommend = {
        "heavytruck": (5e-4, 1e-2),
    }
    if not args.disable_recommended_params:
        args.temp_sample = temp_recommend.get(args.env_name, args.temp_sample)
        args.Ndiffuse = Ndiffuse_recommend.get(args.env_name, args.Ndiffuse)
        args.Nsample = Nsample_recommend.get(args.env_name, args.Nsample)
        args.Hsample = Hsample_recommend.get(args.env_name, args.Hsample)
        beta_override = beta_recommend.get(args.env_name, (args.beta0, args.betaT))
        args.beta0, args.betaT = beta_override
        print(f"override temp_sample to {args.temp_sample}")
    env = mbd.envs.get_env(args.env_name)
    Nx = env.observation_size
    Nu = env.action_size
    # env functions
    # jax.jit compiles to XLA for speed; keep signatures static for compilation.
    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    # rollout_us will itself be vmapped later; we jit the inner function here.
    rollout_us = jax.jit(functools.partial(mbd.utils.rollout_us, step_env_jit))

    rng, rng_reset = jax.random.split(rng)  # NOTE: rng_reset should never be changed.
    state_init = reset_env_jit(rng_reset)

    ## run diffusion

    # 1) Build diffusion noise schedule:
    #    - betas: linear noise increments (higher beta = more noise at that step).
    #    - alphas_bar: cumulative product of (1 - beta), how much "signal" remains.
    #    - sigmas: marginal std of q_t (used to sample noisy actions at step t).
    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)
    Sigmas_cond = (
        (1 - alphas) * (1 - jnp.sqrt(jnp.roll(alphas_bar, 1))) / (1 - alphas_bar)
    )
    sigmas_cond = jnp.sqrt(Sigmas_cond)
    sigmas_cond = sigmas_cond.at[0].set(0.0)
    print(f"init sigma = {sigmas[-1]:.2e}")

    # 2) Start at a prior action trajectory; fall back to zeros if none provided.
    if hasattr(env, "warm_start_actions"):
        try:
            YN = env.warm_start_actions(args.Hsample)
        except Exception:
            YN = jnp.zeros([args.Hsample, Nu])
    else:
        YN = jnp.zeros([args.Hsample, Nu])

    @jax.jit
    def reverse_once(carry, unused):
        """One reverse step at noise level i -> i-1."""
        i, rng, Ybar_i = carry
        # Yi is the current noisy sample drawn from q_i.
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        # 3a) Sample noisy action trajectories from q_i at noise level i.
        rng, Y0s_rng = jax.random.split(rng)
        eps_u = jax.random.normal(Y0s_rng, (args.Nsample, args.Hsample, Nu))
        Y0s = eps_u * sigmas[i] + Ybar_i
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        # 3b) Roll out each trajectory to obtain rewards; keep pipeline states.
        # jax.vmap batches over the Nsample dimension; state_init is shared,
        # Y0s is per-sample. This is equivalent to a for-loop but vectorized.
        rewss, qs = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
        rews = rewss.mean(axis=-1)
        # Normalize returns for a stable softmax; floor std to avoid div-by-0.
        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
        rew_mean = rews.mean()
        logp0 = (rews - rew_mean) / rew_std / args.temp_sample

        # 3c) Optionally boost weights using demo likelihoods (if provided by env).
        #      env.eval_xref_logpd returns a log-likelihood of matching a demo
        #      reference trajectory; we keep whichever weight (reward or demo)
        #      is larger per sample.
        if args.enable_demo:
            xref_logpds = jax.vmap(env.eval_xref_logpd)(qs)
            xref_logpds = xref_logpds - xref_logpds.max()
            logpdemo = (
                (xref_logpds + env.rew_xref - rew_mean) / rew_std / args.temp_sample
            )
            demo_mask = logpdemo > logp0  # prefer demo weight where higher
            logp0 = jnp.where(demo_mask, logpdemo, logp0)
            logp0 = (logp0 - logp0.mean()) / logp0.std() / args.temp_sample

        # 3d) Softmax over trajectories -> weighted mean is the score proxy.
        #     Higher reward (or demo weight) -> larger weight in the mean action.
        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)  # NOTE: update only with reward

        # 3e) Reverse/noise-reduction step using the reward-weighted score proxy.
        #     Here the score proxy approximates the gradient of log likelihood
        #     of good trajectories; the update reduces noise toward high-reward modes.
        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)

        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])

        return (i - 1, rng, Ybar_im1), rews.mean()

    # 3) Run the full reverse diffusion chain.
    def reverse(YN, rng):
        Yi = YN
        Ybars = []
        with tqdm(range(args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                carry_once = (i, rng, Yi)
                (i, rng, Yi), rew = reverse_once(carry_once, None)
                Ybars.append(Yi)
                # Update the progress bar's suffix to show the current reward
                pbar.set_postfix({"rew": f"{rew:.2e}"})
        return jnp.array(Ybars)

    rng_exp, rng = jax.random.split(rng)
    Ybars = reverse(YN, rng_exp)
    if not args.not_render:
        path = f"{mbd.__path__[0]}/../results/{args.env_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        jnp.save(f"{path}/mu_0ts.npy", Ybars)
        if args.env_name == "car2d":
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            # rollout
            xs = jnp.array([state_init.pipeline_state])
            state = state_init
            for t in range(Ybars.shape[1]):
                state = step_env_jit(state, Ybars[-1, t])
                xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
            env.render(ax, xs)
            if args.enable_demo:
                ax.plot(env.xref[:, 0], env.xref[:, 1], "g--", label="RRT path")
            ax.legend()
            plt.savefig(f"{path}/rollout.png")
        elif args.env_name in ["heavytruck", "heavy_truck"]:
            fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
            rewss_final, qs_final = rollout_us(state_init, Ybars[-1])
            xs = jnp.concatenate([state_init.pipeline_state[None], qs_final], axis=0)
            env.render(axes, xs, Ybars[-1], warm_start=YN)
            plt.tight_layout()
            plt.savefig(f"{path}/rollout.png")
            plt.close(fig)

            # Intermediate diffusion snapshots to visualize optimization progress.
            snapshot_ids = jnp.linspace(
                0, Ybars.shape[0] - 1, num=jnp.minimum(4, Ybars.shape[0])
            ).astype(int)
            for idx in snapshot_ids.tolist():
                fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
                _, qs_stage = rollout_us(state_init, Ybars[idx])
                xs = jnp.concatenate([state_init.pipeline_state[None], qs_stage], axis=0)
                env.render(axes, xs, Ybars[idx], warm_start=YN)
                plt.tight_layout()
                plt.savefig(f"{path}/rollout_stage_{idx}.png")
                plt.close(fig)
        else:
            render_us = functools.partial(
                mbd.utils.render_us,
                step_env_jit,
                env.sys.tree_replace({"opt.timestep": env.dt}),
            )
            webpage = render_us(state_init, Ybars[-1])
            with open(f"{path}/rollout.html", "w") as f:
                f.write(webpage)
    rewss_final, qs_final = rollout_us(state_init, Ybars[-1])
    rew_final = rewss_final.mean()

    return rew_final


if __name__ == "__main__":
    rew_final = run_diffusion(args=tyro.cli(Args))
    print(f"final reward = {rew_final:.2e}")
