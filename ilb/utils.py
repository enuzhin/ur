import numpy as np
import torch


def divergence(p, log_prob_prime, log_pi_prime, v, div_v):
    return p * (((log_prob_prime + log_pi_prime) * v).sum(-1) + div_v)


def ensured_probability_density(p, dx):
    p = p.clip(0, None)
    p = p / (p.sum() * np.prod(dx))
    return p


def evaluate_policy(policy, env, gamma, num_steps = 1000, num_episodes = 10):
    return_history = []
    for episode in range(num_episodes):
        episode_return = play_episode(policy,env,gamma, num_steps)
        return_history.append(episode_return)
    mean_return = np.mean(return_history)
    std_return = np.std(return_history)
    return mean_return,std_return


def play_episode(policy,env,gamma, num_steps = 1000):
    state = env.reset()
    reward_history = []
    device = next(policy.parameters()).device
    for t in range(num_steps):
        action = policy.predict(torch.tensor(state).to(device), deterministic=True)
        state,reward,_,_ = env.step(action)
        reward_history.append(reward)

    episode_return = 0
    gamma_dt = gamma**env.dt
    for reward in reversed(reward_history):
        episode_return = gamma_dt * episode_return + reward * env.dt
    return episode_return


def update_state_distribution(p, v_mean, p0, t, dx, dv, dt, N):
    v_mean = torch.moveaxis(v_mean, -1, 0)
    f = p[np.newaxis, :] * v_mean
    div = discrete_divergence(f, (dx, dv))
    p_dot = (p0 - p) / t - div

    p = p + p_dot * dt

    p = p.clip(0, None)
    p = p / (p.sum() * dx * dv)
    p = ensured_reflection(p, N)
    return p


def discrete_divergence(f, dx, num_dims=None):
    if num_dims is None:
        num_dims = len(f)
    return sum([torch.gradient(f[i], dim=i, edge_order=1)[0] / dx[i] for i in range(num_dims)])


def ensured_reflection(p, N):
    f1 = torch.flip(p[0, N // 2:], dims=[0])
    f2 = torch.flip(p[-1, N // 2:], dims=[0])
    p[0, :N // 2 + 1] = f1[:] = (p[0, :N // 2 + 1] + torch.flip(p[0, N // 2:], dims=[0])) / 2
    p[-1, :N // 2 + 1] = f2[:] = (p[-1, :N // 2 + 1] + torch.flip(p[-1, N // 2:], dims=[0])) / 2

    p[:, 0] = p[:, 1]
    p[:, -1] = p[:, -2]
    return p
