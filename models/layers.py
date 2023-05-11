import torch
from torch import nn
import utils
import torch.nn.functional as F


# Decoder for decoding latent space data back to the data space
class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(Decoder, self).__init__()
        decoder = nn.Sequential(nn.Linear(latent_dim, input_dim),)
        utils.init_weight(decoder)
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(x)


# Customized RNN basic unit
class custom_GRU(nn.Module):
    def __init__(self, latent_dim, input_dim, device,
                 update_gate=None,
                 reset_gate=None,
                 new_state_net=None,
                 n_units=10):
        super(custom_GRU, self).__init__()
        if update_gate is None:
            self.update_gate = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim),
                nn.Sigmoid())
            utils.init_weight(self.update_gate)
        else:
            self.update_gate = update_gate

        if reset_gate is None:
            self.reset_gate = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim),
                nn.Sigmoid())
            utils.init_weight(self.reset_gate)
        else:
            self.reset_gate = reset_gate

        if new_state_net is None:
            self.new_state_net = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim * 2)
            )
            utils.init_weight(self.new_state_net)
        else:
            self.new_state_net = new_state_net


    def forward(self, y_mu, y_std, x, masked_update=True):
        y_concat = torch.cat([y_mu, y_std, x], -1)
        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mu * reset_gate, y_std * reset_gate, x], -1)
        new_state, new_state_std = utils.split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()
        new_y = (1 - update_gate) * new_state + update_gate * y_mu
        new_y_std = (1 - update_gate) * new_state_std + update_gate * y_std
        assert(not torch.isnan(new_y).any()), "Nan was found inside the output !"
        # Assume x contains data and mask, update only the states in hidden layer if at least one feature is present for the current time point
        if masked_update:
            n_data_dim = x.size() // 2
            mask = x[:, :, n_data_dim:]
            utils.check_mask(x[:, :, :n_data_dim], mask)
            assert (not torch.isnan(mask).any()), "Nan found in mask tensor, please check!"
            new_y = mask * new_y + (1 - mask) * y_mu
            new_y_std = mask * new_y_std + (1 - mask) * y_std
            if torch.isnan(new_y).any():
                print("There is NaN found in generated datapoints! Please check the following:")
                print(mask)
                print(y_mu)
                print(prev_new_y)
                exit()
        new_y_std = new_y_std.abs()
        return new_y, new_y_std


# RNN with ODE func to retrieve missing time information
class Encoder_odernn(nn.Module):
    def __init__(self, latent_dim, input_dim, device, z0_solver=None,
                 z0_dim=None, gru_update=None, n_gru=10):
        super(Encoder_odernn, self).__init__()

        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim

        if gru_update is None:
            self.gru_update = custom_GRU(latent_dim, input_dim, device=device, n_units=n_gru).to(device)
        else:
            self.gru_update = gru_update

        self.z0_solver = z0_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.extra_info = None

        self.transform_z0 = nn.Sequential(
            nn.Linear(latent_dim * 2, 100),
            nn.Tanh(),
            nn.Linear(100, self.z0_dim * 2))
        utils.init_weight(self.transform_z0)


    def forward(self, data, tps, run_backwards=True, save_info=False):
        """
        Data here should be already concatenated with its mask!
        """
        assert (not torch.isnan(data).any()), "NaN tensor was found inside the data!"
        assert (not torch.isnan(tps).any()), "NaN tensor was found inside the time steps!"
        n_traj, n_tps, n_dim = data.size()

        if len(tps) == 1:
            prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
            prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
            xi = data[:, 0, :].unsqueeze(0)
            last_yi, last_yi_std = self.gru_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            last_yi, last_yi_std, _, extra_info = self.forward_ode(data, tps, run_backwards=run_backwards, save_info=save_info)

        z0_mu =  last_yi.reshape(1, n_traj, self.latent_dim)
        z0_std = last_yi_std.reshape(1, n_traj, self.latent_dim)
        z0_mu, z0_std = utils.split_last_dim(self.transform_z0(torch.cat([z0_mu, z0_std], -1)))
        z0_std = z0_std.abs()
        if save_info:
            self.extra_info = extra_info
        return z0_mu, z0_std


    def forward_ode(self, data, tps, run_backwards=True, save_info=False):
        """
        Data here should be already concatenated with its mask!
        """
        n_traj, n_tps, n_dim = data.size()
        extra_info = []
        t0 = tps[-1]
        if run_backwards:
            t0 = tps[0]
        device = utils.get_device(data)
        prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        prev_t, ti = tps[-1] + 0.01, tps[-1] # 0.01 is added here, should I use 1 (s)?
        interval_length = tps[-1] - tps[0]
        min_step = interval_length / 50
        assert (not torch.isnan(data).any()), "NaN tensor was found inside the data!"
        assert (not torch.isnan(tps).any()), "NaN tensor was found inside the time steps!"
        latent_ys = []

        # Run ODE and backward to combine the y(t) estimated using gating
        tps_iter = range(0, len(tps))
        if run_backwards:
            tps_iter = reversed(tps_iter)
        for i in tps_iter:
            if (prev_t - ti) < min_step:
                tp = torch.stack([prev_y, ti])
                inc = self.z0_solver.ode_func(prev_t, prev_y) * (ti - prev_t)
                assert (not torch.isnan(inc).any()), "Nan values found in the output of ODE solver!, i.e. Nan is increased after time passed"
                ode_sol = prev_y + inc
                ode_sol = torch.stack([prev_y, ode_sol], 2).to(device)
                assert(not torch.isnan(ode_sol).any()), "NaN values found inside the concatenation of T-1 data and T data, consider ODE func issue"
            else:
                n_intermediate_tp = max(2, ((prev_t - ti) / min_step).int())
                tp = utils.linspace_vector(prev_t, ti, n_intermediate_tp)
                ode_sol = self.z0_solver(prev_y, tp)
                assert(not torch.isnan(ode_sol).any()), "The output of ODE solver is NaN found within a minimal time increase!"

            if torch.mean(ode_sol[:, :, 0, :] - prev_y) >= 0.001:   # Consider not using 0.001 but 1?
                print("Error found --> 1st datapoint of ODE is not equal to the initial value")
                print(torch.mean(ode_sol[:, :, 0, :] - prev_y))
                exit()

            yi_ode = ode_sol[:, :, -1, :]
            xi = data[:, i, :].unsqueeze(0)
            yi, yi__std = self.gru_update(yi_ode, prev_std, xi)
            prev_y, prev_std = yi, yi__std
            prev_t, ti = tps[i], tps[i - 1]
            latent_ys.append(yi)

            if save_info:
                d = {
                    "yi_ode": yi_ode.detach(),
                    "yi": yi.detach(),
                    "yi_std": yi.std.detach(),
                    "time_pts": tps.detach(),
                    "ode_sol": ode_sol.detach()}
                extra_info.append(d)

        latent_ys = torch.stack(latent_ys, 1)
        assert (not torch.isnan(yi).any()), "NaN tensor was found in the y initial state!"
        assert (not torch.isnan(yi__std).any()), "NaN tensor was found in the y_std initial state!"
        return yi, yi__std, latent_ys, extra_info



