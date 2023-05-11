import os
import random

import SimpleITK as sitk
import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils import normalize_masked_data, split_n_subsample_batch

matplotlib.use('Agg')


def get_data_minmax(voxels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dmin, dmax = None, None
    inf = torch.Tensor([float("Inf")])[0].to(device)
    for b, (tp, val, mask) in enumerate(voxels):
        batch_min = []
        batch_max = []
        non_missing_vals = val[:, 0][mask[:, 0] == 1]
        if len(non_missing_vals) == 0:
            batch_min.append(inf)
            batch_max.append(-inf)
        else:
            batch_min.append(torch.min(non_missing_vals))
            batch_max.append(torch.max(non_missing_vals))
        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)
        if (dmin is None) and (dmax is None):
            dmin = batch_min
            dmax = batch_max
        return dmin, dmax


def var_time_collate_fn(batch, args, device, data_type="train", dmin=None, dmax=None):
    """
    The shape of one batch is in a tuple of (time points, data points, masks), where
    time points = [L]
    data points = [L, 1]
    masks = [L, 1], where L is the length of the sequence (number of time points observed per one single voxel)

    Original paper normalized the input data (MinMax-scale) and time points (within 0-1s)
    I'm afraid this may destory the Intensity dynamic (since normalized data can be in negative), therefore I did not normalize it
    (need verify) The original data should be already in a "normal distribution"
    """
    D = batch[0][1].shape[1]  # == n_dim == 1
    combined_tp, inverse_idx = torch.unique(torch.cat([ex[0] for ex in batch]), sorted=True, return_inverse=True)
    combined_tp = combined_tp.to(device)
    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tp), D]).to(device)
    combined_masks = torch.zeros([len(batch), len(combined_tp), D]).to(device)
    for b, (tp, val, mask) in enumerate(batch):
        tp = tp.to(device)
        val = val.to(device)
        mask = mask.to(device)
        indices = inverse_idx[offset:offset + len(tp)]
        offset += len(tp)
        combined_vals[b, indices] = val
        combined_masks[b, indices] = mask

    # combined_vals, _, _ = normalize_masked_data(combined_vals, combined_masks, att_min=dmin, att_max=dmax)
    # if torch.max(combined_tp) != 0:
    #     combined_tp = combined_tp / torch.max(combined_tp)

    data_dict = {
        "data": combined_vals,
        "time_steps": combined_tp,
        "mask": combined_masks
    }
    data_dict = split_n_subsample_batch(data_dict, args, data_type=data_type)
    return data_dict


class Image2Signal:
    def __init__(self, patient_id, save_path, cuda):
        self.name = patient_id
        self.img_dir = os.path.join(r"A:\breast_ftv\code_test\image", self.name)
        self.time_dir = os.path.join(r"A:\breast_ftv\code_test\time_info", f"{self.name}_time.csv")
        self.device = cuda
        self.path = save_path
        self._gettime()
        self._getimage()

    def execute(self) -> None:
        def get_arrray(dir):
            img = sitk.ReadImage(dir)
            img = sitk.Cast(img, sitk.sitkFloat32)
            return sitk.GetArrayFromImage(img)

        image_list = [get_arrray(os.path.join(self.img_dir, f'{tag}.mha')) for tag in self.img_tag]
        breast_mask = sitk.ReadImage(os.path.join(self.img_dir, 'VOI_mask.mha'))
        flatten_image_list = [image_list[i][sitk.GetArrayFromImage(breast_mask) == 1] for i in range(len(image_list))]
        data_point = torch.tensor(np.expand_dims(np.dstack(tuple(flatten_image_list)).squeeze(0), -1),
                                  dtype=torch.float32, device=self.device)
        time_point = np.ceil(np.asarray(self.tp_value))
        time_point = np.full((data_point.shape[0], time_point.shape[0]), time_point)
        time_point = torch.tensor(time_point, dtype=torch.float32, device=self.device)
        time_point[:, 0] = torch.zeros_like(time_point[:, 0], dtype=torch.float32, device=self.device)
        voxels = []
        for i in tqdm(range(time_point.shape[0]), total=len(time_point)):
            time_step, val = time_point[i], data_point[i]
            mask = torch.ones_like(val, dtype=torch.float32, device=self.device)
            voxels.append((time_step, val, mask))
        torch.save(voxels, self.path)


    def _gettime(self):
        time_table = pd.read_csv(self.time_dir)
        key = time_table.loc[:, 'name_tag']
        value = time_table.loc[:, 'acquisition_time_difference(s)']
        self.time_dict = dict(zip(key, value))

    def _getimage(self):
        self.img_tag = list(self.time_dict.keys())
        self.tp_value = list(self.time_dict.values())


class preprocessed_data(object):
    def __init__(self, image_dir, raw_dir, device):

        '''
        :param dir: Raw data directory, each file contains .mha(s) for one patient
        :param device: load on gpu or cpu
        :param raw_dir: stored location of pre-processed data
        '''

        self.dir = image_dir
        self.device = device
        self.params = ['Intensity']
        self.location = raw_dir

    def load_data(self, num_patient=None, n_sample=None) -> None:
        self.dataset = []
        if num_patient is None:
            for file in os.listdir(self.location):
                path = os.path.join(self.location, file)
                voxels_from1sample = torch.load(path, map_location=self.device)
                self.dataset += voxels_from1sample
        else:
            for i, file in enumerate(os.listdir(self.location)):
                if i < num_patient:
                    voxels_from1sample = torch.load(os.path.join(self.location, file), map_location=self.device)
                    self.dataset += voxels_from1sample
                else:
                    print(f"voxels of {num_patient} patients has been loaded")
        if n_sample is not None:
            self.dataset = random.sample(self.dataset, n_sample)

    def forward(self) -> None:
        for patient in os.listdir(self.dir):
            if os.path.exists(os.path.join(self.location, f"{patient}.pt")):
                print(f"Patient {patient} has been conveted to the raw data format, going next")
                continue
            else:
                generator = Image2Signal(patient, save_path=os.path.join(self.location, f"{patient}.pt"),
                                         cuda=self.device)
                generator.execute()
                print(f"Done for patient {patient}")
        print("Raw data conversion is done!")

    def visualize(self, t_steps, data, mask, plot_name):
        non0_attr = (torch.sum(mask, 0) > 2).numpy()
        non0idx = [i for i in range(len(non0_attr)) if non0_attr[i] == 1.]
        n_non0 = sum(non0_attr)
        mask = mask[:, non0idx]
        data = data[:, non0idx]
        pass

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    dir = r"A:\breast_ftv\code_test\image"
    raw = r"A:\breast_ftv\TEcode\data\raw_data"

    x = preprocessed_data(dir, raw, device='cuda:0')
    x.forward()
