import os

import pandas as pd
import pydicom
from tqdm import tqdm




class SinglePx_metadata:
    def __init__(self, patient_id):
        self.patient = patient_id
        self.raw_path = r"T:\Online_Image_Source\ISPY2\ISPY2"
        self.metadata = {
            'patient_ID': [patient_id],
            'b0': None,
            'TE': None,
            'TR': None,
            'flip_angle': None,
            'bandwidth': None,
            'Intensity_max': None,
            'Intensity_min': None,
        }

    def export(self) -> None:
        out  =pd.DataFrame.from_dict(self.metadata)
        path =  os.path.join(r"A:\breast_ftv\TEcode\data\meta_data", f"{self.patient}.csv")
        out.to_csv(path)
        print(f"{self.patient}'s metadata have been saved !")


    def fill_data_except_intensity(self) -> None:
        idx = 0
        for dcm in os.listdir(self.target_dir):
            if idx == 0:
                dcm_data = pydicom.dcmread(os.path.join(self.target_dir, dcm))
                try:
                    self.metadata['b0'] = [float(dcm_data.MagneticFieldStrength)]
                    self.metadata['TE'] = [float(dcm_data.EchoTime)]
                    self.metadata['TR'] = [float(dcm_data.RepetitionTime)]
                    self.metadata['flip_angle'] = [float(dcm_data.FlipAngle)]
                    self.metadata['bandwidth'] = [float(dcm_data.PixelBandwidth)]
                except TypeError:
                    print("Unknown Data type found in DICOM, should be NONE")
                    break


                idx += 1
            else:
                break

    def fill_min_max_intensity(self) -> None:
        min_list = []
        max_list = []
        for dcm in os.listdir(self.target_dir):
            dcm_data = pydicom.dcmread(os.path.join(self.target_dir, dcm))
            min_list.append(float(dcm_data.SmallestImagePixelValue))
            max_list.append(float(dcm_data.LargestImagePixelValue))
        min_value = min(min_list)
        max_value = max(max_list)
        self.metadata['Intensity_max'] = [max_value]
        self.metadata['Intensity_min'] = [min_value]






    def get_target_dir(self) -> None:
        patient_folder = os.path.join(self.raw_path, f"ISPY2-{self.patient}")
        target_tp_folder = None
        for files in os.listdir(patient_folder):
            if "ISPY2MRIT0" in files:
                target_tp_folder = os.path.join(patient_folder, files)
            else:
                target_tp_folder = target_tp_folder
        assert target_tp_folder is not None, "The T0 MRI cannot be found!"
        target_dir = None
        for files in os.listdir(target_tp_folder):
            if "original DCE" in files:
                target_dir = os.path.join(target_tp_folder, files)
            else:
                target_dir = target_dir
        assert target_dir is not None, "original DCE DICOM cannot be found!"
        self.target_dir = target_dir


if __name__ == '__main__':
    # dir = r"T:\Online_Image_Source\ISPY2\ISPY2\ISPY2-100899\10-26-2002-100899T0-ISPY2MRIT0-88595\51800.000000-ISPY2 VOLSER uni-lateral cropped original DCE-30523\1-640.dcm"
    # data = pydicom.dcmread(dir)
    raw_path = r"T:\Online_Image_Source\ISPY2\ISPY2"
    label_reference_path = r"T:\Radiomics_Projects\ISPY-2\TE\cleaned_dataset_ISPY2"

    for name in tqdm(os.listdir(label_reference_path), total=len(next(os.walk(label_reference_path))[1])):
        target_path = os.path.join(raw_path, f"ISPY2-{name}")
        if os.path.exists(target_path):
            obj = SinglePx_metadata(name)
            obj.get_target_dir()
            obj.fill_data_except_intensity()
            obj.fill_min_max_intensity()
            out = obj.export()
        else:
            print(f"The raw data of patient {name} cannot be found in the raw data directory! going next")
            continue




