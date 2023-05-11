import torch
from torch import nn
from torchdiffeq import odeint
import utils

"""
NeuralODE constructor, served as the approximating the derivative of real dynamic in ODE Solve 
"""
class DEFunc(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_func_net, device):
        super(DEFunc, self).__init__()
        self.input_dim = input_dim
        self.device = device
        utils.init_weight(ode_func_net)
        self.gradient_net = ode_func_net

    def forward(self, t_current, y_current, backwards=False):
        """
        :param t_current: the current time point
        :param y_current: the data value of current time point
        :param backwards: bool
        :return: the derivative of y w.r.t time (dy/dt)
        """
        grad = self.get_ode_gradient(t_current, y_current)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient(self, t_current, y_current):
        return self.gradient_net(y_current)

    def sample_next_pt_from_prior(self, t_current, y_current):
        return self.get_ode_gradient(t_current, y_current)



"""
Implementation of the vanilla NeuralODE solver from torchdiffeq
"""
class DESolver(nn.Module):
    def __init__(self, input_dim, ode_func, method, latents,
                 device, odeint_rtol=1e-4, odeint_atol=1e-5):
        super(DESolver, self).__init__()
        self.ode_method = method
        self.latents = latents
        self.device = device
        self.ode_func = ode_func
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_pt, t_steps2predict, backwards=False):
        """
        Decoding the latent trajectory (along t_span) with ODESolve
        """
        n_traj_samples, n_traj = first_pt.size()[0], first_pt.size()[1]
        n_dim = first_pt.size()[-1]
        pred_y = odeint(self.ode_func, first_pt, t_steps2predict, rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        pred_y = pred_y.permute(1, 2, 0, 3)

        assert(torch.mean(pred_y[:, :, 0, :] - first_pt) < 0.001), f"The predicted initial value of ODE does not match with the real initial value!"
        assert(pred_y.size()[0] == n_traj_samples), f"The predicted value does not match with the original number of shooting {n_traj_samples}, got{pred_y.size()[0]}"
        assert (pred_y.size()[1] == n_traj), f"The predicted value does not match with the original number of samples {n_traj}, got{pred_y.size()[1]}"
        return pred_y

    def sample_traj_from_prior(self, encoded_starting_pt, t_steps2predict, n_traj_samples=1):
        """
        Decoding latent trajectory using prior samples
        :param encoded_starting_pt:
        :param t_steps2predict: time steps at when to sample the new trajectory
        """
        func = self.ode_func.sample_next_pt_from_prior
        pred_y = odeint(func, encoded_starting_pt, t_steps2predict, rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        pred_y = pred_y.permute(1, 2, 0, 3)
        return pred_y





