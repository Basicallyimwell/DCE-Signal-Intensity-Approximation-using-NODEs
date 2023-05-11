import json

from easydict import EasyDict
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

import utils
from data.raw_data_gen import preprocessed_data, var_time_collate_fn, get_data_minmax
from models.DEsolvers import DEFunc, DESolver
from models.blocks import Latent_ODE
from models.layers import Encoder_odernn, Decoder

"""
Creation the dataset object used from raw data to model readable format
"""


def parse_datasets(args, device):
    # def basic_collate_fn(batch, time_steps, args=args, device=device, data_type="train"):
    #     batch = torch.stack(batch)
    #     data_dict = {'data': batch,
    #                  'time_steps': time_steps}
    #     data_dict = utils.split_n_subsample_batch(data_dict, args=args, data_type=data_type)
    #     return data_dict

    # n_total_tp = args.n_tps + args.extrapolate  # extrapolate should be boolean??
    # max_t_extrapolate = args.max_tspan / args.n_tps * n_total_
    train_obj = preprocessed_data(args.image_dir, args.raw_dir, device=device)
    train_obj.load_data(num_patient=args.n_patients, n_sample=args.n_sample)
    train_data, test_data = train_test_split(train_obj.dataset, train_size=0.7, random_state=args.seed, shuffle=True)
    tps, vals, masks = train_data[0]
    n_samples = len(train_obj)
    input_dims = vals.size(-1)
    batch_size = min(min(len(train_obj), args.batch_size), len(train_obj))
    d_min, d_max = get_data_minmax(train_obj)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                  collate_fn=lambda batch: var_time_collate_fn(batch, args, device, data_type="train",
                                                                               dmin=d_min, dmax=d_max))
    test_dataloader = DataLoader(test_data, batch_size=n_samples, shuffle=False,
                                 collate_fn=lambda batch: var_time_collate_fn(batch, args, device, data_type="test",
                                                                              dmin=d_min, dmax=d_max))
    data_obj = {"dataset": train_obj,
                "train_loader": utils.inf_generator(train_dataloader),
                "test_loader": utils.inf_generator(test_dataloader),
                "input_dim": input_dims,
                "n_train_batches": len(train_dataloader),
                "n_test_batches": len(test_dataloader),
                }

    return data_obj


"""
Creating the generative model
"""


def parse_model(args, input_dim, z0_prior, obs_std, device, classif_per_tp=False):
    dim = args.latent_dim
    if args.poisson:
        raise NotImplementedError
    else:
        dim = args.latent_dim
        ode_func_net = utils.create_net(dim, args.latent_dim,
                                        n_layers=args.num_decnode_layers, n_units=args.node_hidden_dim,
                                        non_linear=nn.Tanh)
        decoder_ode_func = DEFunc(input_dim=input_dim, latent_dim=args.latent_dim, ode_func_net=ode_func_net,
                                  device=device).to(device)

    z0_solver = None
    enc_dim = args.encoder_dim
    enc_input_dim = int(input_dim) * 2
    dec_dim = input_dim
    z0_dim = args.latent_dim
    if args.poisson:
        raise NotImplementedError
    ode_func_net = utils.create_net(enc_dim, enc_dim,
                                    n_layers=args.num_encnode_layers, n_units=args.node_hidden_dim, non_linear=nn.Tanh)
    encoder_ode_func = DEFunc(input_dim=enc_input_dim, latent_dim=enc_dim, ode_func_net=ode_func_net, device=device).to(
        device)
    z0_solver = DESolver(enc_input_dim, encoder_ode_func, args.enc_method, args.latent_dim,
                         odeint_rtol=args.rtol, odeint_atol=args.atol, device=device)
    z0_encoder = Encoder_odernn(enc_dim, enc_input_dim, device=device, z0_solver=z0_solver,
                                z0_dim=z0_dim, n_gru=args.num_rnn).to(device)

    latent_decoder = Decoder(args.latent_dim, dec_dim).to(device)
    latent_solver = DESolver(dec_dim, decoder_ode_func, args.dec_method, args.latent_dim,
                             odeint_rtol=args.rtol, odeint_atol=args.atol, device=device)

    model = Latent_ODE(
        input_dim=dec_dim,
        latent_dim=args.latent_dim,
        encoder_z0=z0_encoder,
        decoder=latent_decoder,
        de_solver=latent_solver,
        z0_prior=z0_prior,
        device=device,
        obs_std=obs_std,
        classif_per_tp=classif_per_tp,
        use_poisson=args.poisson,
        train_w_recon=args.train_w_recon
    ).to(device)
    return model


if __name__ == "__main__":
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        dic = json.load(f)
        args = EasyDict(dic)

    x = parse_datasets(args, device='cuda:0')
    in_dim = x["input_dim"]
    batch_dict = utils.next_batch(x["train_loader"])
