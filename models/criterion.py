from torch.distributions import Independent
from torch.distributions.normal import Normal

from utils import *


def mse(mu, data, idx=None):
    n_pts = mu.size()[-1]
    if n_pts > 0:
        loss = nn.MSELoss()
    else:
        loss = torch.zeros([1]).to(get_device(data)).squeeze()
    return loss


def compute_masked_likelihood(mu, data, mask, f):
    # Compute likelihood per voxel and per attr(Intensity, 1)
    n_traj_samples, n_traj, n_tps, n_dim = data.size()
    res = []  # [n_traj * n_traj_samples, 1]
    for i in range(n_traj_samples):
        for k in range(n_traj):
            for j in range(n_dim):
                data_masked = torch.masked_select(data[i, k, :, j], mask[i, k, :, j].bool())
                mu_masked = torch.masked_select(mu[i, k, :, j], mask[i, k, :, j].bool())
                log_prob = f(mu_masked, data_masked, idx=(i, j, k))
                res.append(log_prob)
    res = torch.stack(res, 0).to(get_device(data))
    res = res.reshape((n_traj_samples, n_traj, n_dim))
    res = torch.mean(res, -1)
    res = res.transpose(0, 1)
    return res


def compute_mse(mu, data, mask=None):
    if (len(mu.size()) == 2):
        mu = mu.unsqueeze(0)
    if (len(data.size()) == 2):
        data = data.unsqueeze(0).unsqueeze(2)
    elif (len(data.size()) == 3):
        data = data.unsqueeze(0)
    n_traj_samples, n_traj, n_tps, n_dim = mu.size()
    assert (data.size()[-1] == n_dim), f"the output does not have the same feature dimension as the input, got {data.size()[-1]}"
    if mask is None:
        mu_flat = mu.reshape(n_traj_samples * n_traj, n_tps * n_dim)
        n_traj_samples, n_traj, n_tps, n_dim = data.size()
        data_flat = data.reshape(n_traj_samples * n_traj, n_tps * n_dim)
        res = mse(mu_flat, data_flat)
    else:
        res = compute_masked_likelihood(mu, data, mask, mse)
    return res


def gaussian_loglikelihood(mu2d, data2d, obs_std, idx=None):
    n_data_pts = mu2d.size()[-1]
    if n_data_pts > 0:
        gaussian = Independent(Normal(loc=mu2d, scale=obs_std.repeat(n_data_pts)), 1)
        log_prob = gaussian.log_prob(data2d)
        log_prob = log_prob / n_data_pts
    else:
        log_prob = torch.zeros([1.]).to(get_device(data2d)).squeeze()
    return log_prob


def masked_gaussian_logdensity(mu, data, obs_std, mask=None):
    if (len(mu.size()) == 3):
        mu = mu.unsqueeze(0)
    if (len(data.size()) == 2):
        data = data.unsqueeze(0).unsqueeze(2)
    elif (len(data.size()) == 3):
        data = data.unsqueeze(0)
    n_traj_samples, n_traj, n_tps, n_dim = mu.size()
    assert (data.size()[-1] == n_dim), f"the output does not have the same feature dimension as the input, got {data.size()[-1]}"
    if mask is None:
        mu_flat = mu.reshape(n_traj_samples * n_traj, n_tps * n_dim)
        n_traj_samples, n_traj, n_tps, n_dim = data.size()
        data_flat = data.reshape(n_traj_samples * n_traj, n_tps * n_dim)
        res = gaussian_loglikelihood(mu_flat, data_flat, obs_std)
        res = res.reshape(n_traj_samples, n_traj).transpose(0, 1)
    else:
        func = lambda mu, data, idx: gaussian_loglikelihood(mu, data, obs_std=obs_std, idx=idx)
        res = compute_masked_likelihood(mu, data, mask, func)
    return res
