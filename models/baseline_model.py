from models.criterion import *


class Baseline(nn.Module):
    def __init__(self, input_dim, latent_dim,
                 device,
                 obs_std=0.01,
                 classif_per_tp=False,
                 use_poisson=False,
                 train_w_recon=False):
        super(Baseline, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.obs_std = torch.Tensor([obs_std]).to(device)
        self.classif_per_tp = classif_per_tp
        self.use_poisson = use_poisson
        self.train_w_recon = train_w_recon

        z0_dim = latent_dim

    # def reconstruction(self):
    #     raise NotImplementedError

    def get_gaussian_likelihood(self, true, pred, mask=None):
        log_density_data = masked_gaussian_logdensity(pred, true, obs_std=self.obs_std, mask=mask)
        log_density_data = log_density_data.permute(1, 0)
        log_density = torch.mean(log_density_data, 0)
        return log_density

    def get_mse(self, true, pred, mask=None):
        if mask is not None:
            mask = mask.repeat(pred.size(0), 1, 1, 1)
        log_density_data = compute_mse(pred, true, mask=mask)
        return torch.mean(log_density_data)

    def get_total_loss(self, batch_dict, n_tp2sample=None, n_traj_samples=1, kl_coef=1.):
        pred_x, info = self.reconstruction(batch_dict["tp2predict"],
                                           batch_dict["observed_data"],
                                           batch_dict["observed_tp"],
                                           batch_dict["observed_mask"],
                                           n_traj_samples=n_traj_samples,
                                           mode=batch_dict["mode"])
        likelihood = self.get_gaussian_likelihood(batch_dict["data2predict"], pred_x, mask=batch_dict["masked_predicted_data"])
        recon_error = self.get_mse(batch_dict["data2predict"], pred_x, mask=batch_dict["masked_predicted_data"])


class VAEbase(nn.Module):
    def __init__(self, input_dim, latent_dim,
                 z0_prior, device,
                 obs_std=0.01,
                 classif_per_tp=False,
                 use_poisson=False,
                 train_w_recon=False):
        super(VAEbase, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.z0_prior = z0_prior
        self.obs_std = torch.Tensor([obs_std]).to(device)
        self.classif_per_tp = classif_per_tp
        self.use_poisson = use_poisson
        self.train_w_recon = train_w_recon

    def get_gaussian_likelihood(self, true, pred, mask=None):
        '''
        :param true: [n_traj_samples, n_traj, n_tp, 1]
        :param pred: [n_traj, n_tp, 1]
        :return:  [n_traj_samples]
        '''
        n_traj, n_tp, n_dim = true.size()
        true_repeated = true.repeat(pred.size(0), 1, 1, 1)
        if mask is not None:
            mask = mask.repeat(pred.size(0), 1, 1, 1)
        log_density_data = masked_gaussian_logdensity(pred, true_repeated, obs_std=self.obs_std, mask=mask)
        log_density_data = log_density_data.permute(1, 0)
        log_density = torch.mean(log_density_data, 1)
        return log_density

    def get_mse(self, true, pred, mask=None):
        true_repeated = true.repeat(pred.size(0), 1, 1, 1)
        if mask is not None:
            mask = mask.repeat(pred.size(0), 1, 1, 1)
        log_density_data = compute_mse(pred, true_repeated, mask=mask)
        return torch.mean(log_density_data)

    def get_total_loss(self, batch_dict, n_traj_samples=1, kl_coef=1.):
        pred_y, info = self.reconstruction(batch_dict["tp2predict"],
                                               batch_dict["observed_data"],
                                               batch_dict["observed_tp"],
                                               mask=batch_dict["observed_mask"],
                                               n_traj_samples=n_traj_samples,
                                               mode=batch_dict["mode"])
        pass
