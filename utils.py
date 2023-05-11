import os
import logging
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import glob
import re
from shutil import copyfile
import subprocess
import datetime





def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


"""
The below part related to sample construction
"""


def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def dict_template():
    return {"observed_data": None,
            "observed_tp": None,
            "data2predict": None,
            "tp2predict": None,
            "observed_mask": None,
            "masked_predicted_data": None}


def next_batch(dataloader):
    data_dict = dataloader.__next__()
    batch_dict = dict_template()
    non_missing_tp = torch.sum(data_dict["observed_data"], (0, 2)) != 0.
    batch_dict["observed_data"] = data_dict["observed_data"][:, non_missing_tp]
    batch_dict["observed_tp"] = data_dict["observed_tp"][non_missing_tp]
    if ("observed_mask" in data_dict) and (data_dict["observed_mask"] is not None):
        batch_dict["observed_mask"] = data_dict["observed_mask"][:, non_missing_tp]
    batch_dict["data2predict"] = data_dict["data2predict"]
    batch_dict["tp2predict"] = data_dict["tp2predict"]
    non_missing_tp = torch.sum(data_dict["data2predict"], (0, 2)) != 0.
    batch_dict["data2predict"] = data_dict["data2predict"][:, non_missing_tp]
    batch_dict["tp2predict"] = data_dict["tp2predict"][non_missing_tp]
    if ("masked_predicted_data" in data_dict) and (data_dict["masked_predicted_data"] is not None):
        batch_dict["masked_predicted_data"] = data_dict["masked_predicted_data"][:, non_missing_tp]
    batch_dict["mode"] = data_dict["mode"]
    return batch_dict



def split_data_extrapolate(data_dict):
    device = get_device(data_dict["data"])
    n_obs_tp = data_dict["data"].size(1) // 2
    split_dict = {"observed_data": data_dict["data"][:, :n_obs_tp, :].clone(),
                  "observed_tp": data_dict["time_steps"][:n_obs_tp].clone(),
                  "data2predict": data_dict["data"][:, n_obs_tp:, :].clone(),
                  "tp2predict": data_dict["time_steps"][n_obs_tp:].clone(),
                  "observed_mask": None,
                  "masked_predicted_data": None}
    if ("mask" in data_dict) and (data_dict["mask"] is not None):
        split_dict["observed_mask"] = data_dict["mask"][:, :n_obs_tp].clone()
        split_dict["masked_predicted_data"] = data_dict["mask"][:, n_obs_tp:].clone()
    split_dict["mode"] = "extrapolate"
    return split_dict


def split_data_interpolate(data_dict):
    device = get_device(data_dict["data"])
    split_dict = {"observed_data": data_dict["data"].clone(),
                  "observed_tp": data_dict["time_steps"].clone(),
                  "data2predict": data_dict["data"].clone(),
                  "tp2predict": data_dict["time_steps"].clone(),
                  "observed_mask": None,
                  "masked_predicted_data": None}
    if ("mask" in data_dict) and (data_dict["mask"] is not None):
        split_dict["observed_mask"] = data_dict["mask"].clone()
        split_dict["masked_predicted_data"] = data_dict["mask"].clone()
    split_dict["mode"] = "interpolate"
    return split_dict


def add_mask(data_dict):
    """
    In the beginning the "observed mask" and "masked_predicted_data" are == "mask"
    Since unsampled time points are added (those corresponding data value is 0),
    these pseudo-datapoints(i.e. 0) are regarded as the observed data points where "observed_mask" == 1
    The "masked_predicted_data" == 1 indicate the real data observation (also purposed for reconstruction prediction)

    In the condition when no additional subsample timepoints and datapoint removal,
    --> split["observed_mask"] == ones_like(split["observed_data"])
    --> split["predicted_masked_data"] == split["observed_mask"]

    In the condition when additional subsample timepoints and datapoint removal are present,
    ...... (To be continued)
    """
    data = data_dict["observed_data"]
    mask = data_dict["observed_mask"]
    if mask is not None:
        mask = torch.ones_like(data).to(get_device(data))
    data_dict["observed_mask"] = mask
    return data_dict


def normalize_masked_data(vals, masks, att_min, att_max):
    att_max[att_max == 0.] = 1.  # No 0 division
    if (att_max != 0.).all():
        vals_norm = (vals - att_min) / att_max
    else:
        raise Exception("Zero found!")
    if torch.isnan(vals_norm).any():
        raise Exception("Nans found!")
    vals_norm[masks == 0] = 0
    return vals_norm, att_min, att_max


def subsample_tp(data, time_steps, mask, n_tp2sample=None):
    if n_tp2sample is None:
        return data, time_steps, mask
    n_tp_in_batch = len(time_steps)
    if n_tp2sample > 1:
        assert (n_tp2sample <= n_tp_in_batch), "number of points to sample cannot be larger than the number of points within this batch !"
        n_tp2sample = int(n_tp2sample)
        for i in range(data.size(0)):
            missing_idx = sorted(
                np.random.choice(np.arange(n_tp_in_batch), (n_tp_in_batch - n_tp2sample), replace=False))
            data[i, missing_idx] = 0.
            if mask is not None:
                mask[i, missing_idx] = 0.
    elif (n_tp2sample <= 1) and (n_tp2sample > 0):
        percent_tp2sample = n_tp2sample
        for i in range(data.size(0)):
            current_mask = mask[i].sum(-1).cpu()
            non_missing_tp = np.where(current_mask > 0)[0]
            n_tp_current = len(non_missing_tp)
            n2sample = int(n_tp_current * percent_tp2sample)
            subsample_idx = sorted(np.random.choice(non_missing_tp, n2sample, replace=False))
            tp2set2zero = np.setdiff1d(non_missing_tp, subsample_idx)
            data[i, tp2set2zero] = 0.
            if mask is not None:
                mask[i, tp2set2zero] = 0.
    return data, time_steps, mask


def cutout_tp(data, time_steps, mask, n_pt2cut=None):
    if n_pt2cut is None:
        return data, time_steps, mask
    n_tp_in_batch = len(time_steps)
    if n_pt2cut < 1:
        raise Exception("Number of timepoints to be cut must larger than 1")
    assert (n_pt2cut <= n_tp_in_batch), "Number of datapoints being cut cannot be larger than the total number of datapoints in this batch!"
    n_pt2cut = int(n_pt2cut)
    for i in range(data.size(0)):
        start = np.random.choice(np.arange(5, (n_tp_in_batch - n_pt2cut - 5)), replace=False)
        data[i, start:(start + n_pt2cut)] = 0.
        if mask is not None:
            mask[i, start:(start + n_pt2cut)] = 0.
    return data, time_steps, mask


def subsample_obs_data(data_dict, n_tp2sample=None, n_pt2cut=None):
    '''
    :param data_dict: split data dict
    :param n_tp2sample:  if not None --> random sampling, return timeline = [n_tp2sample]
    :param n_pt2cut:  if not None --> cut datapoint consecutively on timeline, return timeline with (N - n_pt2cut) points
    :return: new data_dict
    '''

    if n_tp2sample is not None:
        data, time_steps, mask = subsample_tp(data_dict["observed_data"].clone(),
                                              time_steps=data_dict["observed_tp"].clone(),
                                              mask=data_dict["observed_mask"].clone() if data_dict[
                                                                                             "observed_mask"] is not None else None,
                                              n_tp2sample=n_tp2sample)

    if n_pt2cut is not None:
        data, time_steps, mask = cutout_tp(data_dict["observed_data"].clone(),
                                           time_steps=data_dict["observed_tp"].clone(),
                                           mask=data_dict["observed_mask"].clone() if data_dict[
                                                                                          "observed_mask"] is not None else None,
                                           n_pt2cut=n_pt2cut)

    new_data_dict = {}
    for key in data_dict.keys():
        new_data_dict[key] = data_dict[key]
    new_data_dict["observed_data"] = data.clone()
    new_data_dict["observed_tp"] = time_steps.clone()
    new_data_dict["observed_mask"] = mask.clone()
    if n_pt2cut is not None:
        new_data_dict["data2predict"] = data.clone()
        new_data_dict["tp2predict"] = time_steps.clone()
        new_data_dict["masked_predicted_data"] = mask.clone()
    return new_data_dict


def split_n_subsample_batch(data_dict, args, data_type="train"):
    if data_type == "train":
        if args.extrapolate:
            processed_dict = split_data_extrapolate(data_dict)
        else:
            processed_dict = split_data_interpolate(data_dict)
    else:
        if args.extrapolate:
            processed_dict = split_data_extrapolate(data_dict)
        else:
            processed_dict = split_data_interpolate(data_dict)

    processed_dict = add_mask(processed_dict)
    if (args.sample_tp is not None) or (args.cut_pt is not None):
        processed_dict = subsample_obs_data(data_dict=processed_dict,
                                            n_tp2sample=args.sample_tp,
                                            n_pt2cut=args.cut_pt)
    return processed_dict


"""
The below part are related to modeling 
"""


def init_weight(net, std=0.1) -> None:
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=std)
            nn.init.constant_(m.bias, val=0.)


def flatten(tensor, dim):
    return tensor.reshape(tensor.size()[:dim] + (-1, ))


def split_last_dim(tensor):
    last_dim = tensor.size()[-1]
    last_dim = last_dim // 2
    res = None
    if len(tensor.size) == 3:
        res = tensor[:, :, :last_dim] , tensor[:, :, last_dim:]
    if len(tensor.size) == 2:
        res = tensor[:, :last_dim], tensor[:, last_dim:]
    return res


def check_mask(data, mask) -> None:
    n_zeros = torch.sum(mask == 0.).cpu().numpy()
    n_ones = torch.sum(mask == 1.).cpu().numpy()
    assert ((n_zeros + n_ones) == np.prod(list(mask.size()))), "Masks should only contains 0. s and 1. s"
    assert (torch.sum(data[mask == 0.] != 0.) == 0), "All elements that being masked out should be 0. !"


def linspace_vector(sos, eos, num_pts):   # start can be either single value or vector
    size = np.prod(sos.size())
    assert(sos.size() == eos.size()), "The size of starting and ending did not match!!"
    if size == 1:
        res = torch.linspace(sos, eos, num_pts)
    else:
        res = torch.Tensor()
        for i in range(0, sos.size(0)):
            res = torch.cat((res, torch.linspace(sos[i], eos[i], num_pts)), 0)
        res = torch.t(res.reshape(sos.size(0), num_pts))
    return res


def shift_outputs(outputs, first_datapoint=None):
    outputs = outputs[:, :, :-1, :]
    if first_datapoint is not None:
        n_traj, n_dim = first_datapoint.size()
        first_datapoint = first_datapoint.reshape(1, n_traj, 1, n_dim)
        outputs = torch.cat([first_datapoint, outputs], 2)
    return outputs


def sample_std_gaussian(mu, std):
    device = get_device(mu)
    d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * std.float() + mu.float()


def create_net(n_input, n_output,
               n_layers=1, n_units=100, non_linear=nn.Tanh):
    """
    :param n_input: input_dim, where should be == latent dim of ODE_RNN
    :param n_output: output_dim, where should be == input_dim of ODE_RNN
    :param n_layers: number of linear layer used for the decoding
    :param n_units: dim of hidden layer used for nn.Linear (converting output of RNN into a mu and sigma)
    :param non_linear:  Activation layer class, default nn.Tanh
    :return: nn.Sequential()
    """
    layers = [nn.Linear(n_input, n_units)]
    for i in range(n_layers):
        layers.append(non_linear())
        layers.append(nn.Linear(n_units, n_units))
    layers.append(non_linear())
    layers.append(nn.Linear(n_units, n_output))
    return nn.Sequential(*layers)
