import torch
from torch import nn
import utils
from models.baseline_model import Baseline, VAEbase
from models.layers import *
import criterion

from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter

"""
The encoder block of the generative model
"""
class ODE_RNN(Baseline):
    def __init__(self, input_dim, latent_dim, device, z0_solver=None, n_gru=10, n_units=100,
                 concat_mask=False, obs_std=0.1, classif_per_tp=False, train_w_recon=False):
        Baseline.__init__(self, input_dim, latent_dim, device=device, obs_std=obs_std, classif_per_tp=classif_per_tp, train_w_recon=train_w_recon)
        encoding_dim = latent_dim

        # The input_dim is (input_dim*2) since we input both vals and masks
        self.ode_rnn = Encoder_odernn(latent_dim=encoding_dim, input_dim=(input_dim) * 2,
                                      device=device, z0_solver=z0_solver, n_gru=n_gru)

        self.z0_solver = z0_solver

        self.decoder = nn.Sequential(nn.Linear(latent_dim, n_units),
                                     nn.Tanh(),
                                     nn.Linear(n_units, input_dim),)
        utils.init_weight(self.decoder)

    def reconstruction(self, tp2predict, observed_data, observed_tp, mask=None,
                       n_traj_samples=None, mode=None):
        if (len(observed_tp) != len(tp2predict)) or (torch.sum(tp2predict - observed_tp) != 0):
            raise Exception("Extrapolation is not implemented (or made) yet!")
        assert(len(observed_tp) == len(tp2predict)), "The number of observed tp does not match with the number of tp2predict"
        assert(mask is not None), "No mask was found!"

        masked_data = observed_data
        if mask is not None:
            masked_data = torch.cat([observed_data, mask], -1)
        _, _, latent_ys, _ = self.ode_rnn(masked_data, observed_tp, run_backwards=False)
        latent_ys = latent_ys.permute(0, 2, 1, 3)
        last_hidden = latent_ys[:, :, -1, :]
        outputs = self.decoder(latent_ys)
        first_pt = observed_data[:, 0, :]
        outputs = utils.shift_outputs(outputs, first_pt)
        extra_info = {
            "first_point": (latent_ys[:, :, -1, :], 0.0, latent_ys[:, :, -1, :])
        }
        ############# Do we need a classification tasks? (currently none)
        return outputs, extra_info


"""
The decoder block of the generative model
"""

class Latent_ODE(VAEbase):
    def __init__(self, input_dim, latent_dim, encoder_z0, decoder, de_solver, z0_prior, device,
                 obs_std=0.1, classif_per_tp=False, use_poisson=False, train_w_recon=False):
        super(Latent_ODE, self).__init__( input_dim=input_dim, latent_dim=latent_dim, z0_prior=z0_prior, device=device,
                                          obs_std=obs_std, classif_per_tp=classif_per_tp, train_w_recon=train_w_recon)
        self.encoder_z0 = encoder_z0
        self.de_solver = de_solver
        self.decoder = decoder
        self.use_poisson = use_poisson

    def reconstruction(self, tp2predict,  observed_data, observed_tp, mask=None,
                       n_traj_samples=None, run_backwards=True, mode=None):

        if isinstance(self.encoder_z0, Encoder_odernn):
            masked_true = observed_data
            if mask is not None:
                masked_true = torch.cat([observed_data, mask], -1)
            first_pt_mu, first_pt_std = self.encoder_z0(masked_true, observed_tp, run_backwards=run_backwards)
            z0_mu = first_pt_mu.repeat(n_traj_samples, 1, 1)
            z0_std = first_pt_std.repeat(n_traj_samples, 1, 1)
            first_pt_encoded = utils.sample_std_gaussian(z0_mu, z0_std)
        else:
            raise Exception(f"Unknown Encoder type == {type(self.encoder_z0).__name__} detected !")

        first_pt_std = first_pt_std.abs()
        assert(torch.sum(first_pt_std < 0) == 0.), "The std of the first latent variables should be take the absolute value!"
        if self.use_poisson:
            raise Exception("Not implemented yet!")
        else:
            first_pt_aug = first_pt_encoded
            aug_z0_mu = z0_mu
        assert(not torch.isnan(tp2predict).any()), "Nan should not be found in tp2predict!"
        assert(not torch.isnan(first_pt_encoded).any()), "Nan should not be found in z0!"
        assert(not torch.isnan(first_pt_aug).any()), "Nan should not be found in augmented z0!"

        sol_y = self.de_solver(first_pt_aug, tp2predict)
        if self.use_poisson:
            raise Exception("Not implemented yet!")
        pred_x = self.decoder(sol_y)
        all_extra_info = {
            "first_point": (first_pt_mu, first_pt_std, first_pt_encoded),
            "latent_traj": sol_y.detach()
        }
        if self.use_poisson:
            raise Exception("Not implemented yet!")

        ############ classification task is needed here (current none)
        return pred_x, all_extra_info

    def sample_traj_from_prior(self, tp2predict, n_traj_samples=1):
        encoded_z0 = self.z0_prior.sample([n_traj_samples, 1, self.latent_dim]).squeeze(-1)
        aug_encoded_z0 = encoded_z0

        if self.use_poisson:
            raise Exception("Not implemented yet!")
        sol_y = self.de_solver.sample_traj_from_prior(aug_encoded_z0, tp2predict, n_traj_samples=3)
        if self.use_poisson:
            raise Exception("Not implemented yet!")

        return self.decoder(sol_y)





