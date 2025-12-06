import jax
from brax.io import html


# evaluate the diffused uss
def eval_us(step_env, state, us):
    def step(carry, u):
        state, done = carry

        def _step(_):
            next_state = step_env(state, u)
            return (next_state, next_state.done), next_state.reward

        def _stay(_):
            return (state, done), 0.0

        return jax.lax.cond(done > 0.0, _stay, _step, None)

    (_, _), rews = jax.lax.scan(step, (state, 0.0), us)
    return rews

def rollout_us(step_env, state, us):
    def step(carry, u):
        state, done = carry

        def _step(_):
            next_state = step_env(state, u)
            return (next_state, next_state.done), (next_state.reward, next_state.pipeline_state)

        def _stay(_):
            # Freeze dynamics once done; keep last state/pipeline for later stats.
            return (state, done), (0.0, state.pipeline_state)

        return jax.lax.cond(done > 0.0, _stay, _step, None)

    (_, _), (rews, pipline_states) = jax.lax.scan(step, (state, 0.0), us)
    return rews, pipline_states


def render_us(step_env, sys, state, us):
    rollout = []
    rew_sum = 0.0
    Hsample = us.shape[0]
    for i in range(Hsample):
        rollout.append(state.pipeline_state)
        state = step_env(state, us[i])
        rew_sum += state.reward
    # rew_mean = rew_sum / (Hsample)
    # print(f"evaluated reward mean: {rew_mean:.2e}")
    return html.render(sys, rollout)
