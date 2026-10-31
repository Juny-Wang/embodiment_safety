"""Explicit SERL sampling contract; deterministic=True is not a vendor API."""
def sample_policy(agent, observations, *, key, deterministic):
    if deterministic:
        return agent.sample_actions(observations=observations, argmax=True)
    return agent.sample_actions(observations=observations, seed=key, argmax=False)
