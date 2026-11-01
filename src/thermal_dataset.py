# -*- coding: utf-8 -*-
"""
thermal_dataset.py
-------------------
PyTorch Dataset for the Squirrel-Cage Induction Motor thermal images
(Zenodo: https://doi.org/10.5281/zenodo.8203070)

Outputs are shaped to plug directly into RadiometricThermalCNN from
thermal_image_module.py:
    - 1-channel, 128x128 tensors
    - 3 broad classes: 0 = Normal Operation, 1 = Shaft Misalignment,
                        2 = Broken Rotor Bars
  (matches MOTOR_CLASSES in thermal_image_module.py)

Folder layout expected (per coupling):
    root_dir/
        current-load-0A/workswell_wic_640/*.png
        misalignment-current-load-0A/workswell_wic_640/*.png
        misalignment-2-current-load-4A/workswell_wic_640/*.png
        rotor-1-current-load-0A/workswell_wic_640/*.png
        ...

Labels come purely from the experiment folder name — filenames are just
numeric timestamps and carry no label information.
"""

import os
import re

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Must match MOTOR_CLASSES in thermal_image_module.py
LABEL_TO_IDX = {
    "normal": 0,
    "misalignment": 1,
    "rotor_fault": 2,
}
IDX_TO_NAME = {
    0: "Normal Operation",
    1: "Shaft Misalignment",
    2: "Broken Rotor Bars",
}


def get_label_from_folder(folder_name: str):
    """
    Map an experiment folder name to (broad_label_str, severity_or_count, current_load_amps).

    Examples:
        current-load-6A                 -> ("normal", None, 6)
        misalignment-current-load-0A    -> ("misalignment", "mild", 0)
        misalignment-2-current-load-4A  -> ("misalignment", "moderate", 4)
        misalignment-3-current-load-2A  -> ("misalignment", "severe", 2)
        rotor-1-current-load-0A         -> ("rotor_fault", "1", 0)
    """
    name = folder_name.lower()

    load_match = re.search(r'(\d+)a', name)
    current_load = int(load_match.group(1)) if load_match else -1

    if 'misalignment-3' in name:
        return "misalignment", "severe", current_load
    if 'misalignment-2' in name:
        return "misalignment", "moderate", current_load
    if 'misalignment' in name:
        return "misalignment", "mild", current_load
    if 'rotor' in name:
        rotor_match = re.search(r'rotor-(\d+)', name)
        n = rotor_match.group(1) if rotor_match else 'unknown'
        return "rotor_fault", n, current_load
    return "normal", None, current_load


def normalize_and_resize(img_raw: np.ndarray, target_size=(128, 128)) -> np.ndarray:
    """16-bit raw thermal array -> normalized [0,1] float32, resized to target_size."""
    img_raw = img_raw.astype(np.float32)
    t_min, t_max = float(img_raw.min()), float(img_raw.max())
    norm = (img_raw - t_min) / max(1e-5, t_max - t_min)
    resized = cv2.resize(norm, target_size, interpolation=cv2.INTER_AREA)
    return resized


class ThermalMotorDataset(Dataset):
    def __init__(self, root_dirs, resolution_folder='workswell_wic_640',
                 target_size=(128, 128), transform=None):
        """
        root_dirs: single path (str) or list of paths, e.g.
            "dataset/thermal_squirrel_cage/thermal_images/coupling_1"
            or
            ["dataset/thermal_squirrel_cage/thermal_images/coupling_1",
             "dataset/thermal_squirrel_cage/thermal_images/coupling_2"]
        """
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]

        self.target_size = target_size
        self.transform = transform
        self.samples = []  # (image_path, broad_label, severity, current_load)

        for root_dir in root_dirs:
            if not os.path.isdir(root_dir):
                print(f"[warning] root_dir not found, skipping: {root_dir}")
                continue

            for exp_folder in sorted(os.listdir(root_dir)):
                exp_path = os.path.join(root_dir, exp_folder)
                img_dir = os.path.join(exp_path, resolution_folder)
                if not os.path.isdir(img_dir):
                    continue

                broad_label, severity, current_load = get_label_from_folder(exp_folder)

                for fname in sorted(os.listdir(img_dir)):
                    if fname.lower().endswith('.png'):
                        self.samples.append(
                            (os.path.join(img_dir, fname), broad_label, severity, current_load)
                        )

        counts = {}
        for _, lbl, _, _ in self.samples:
            counts[lbl] = counts.get(lbl, 0) + 1

        print(f"Total samples: {len(self.samples)}")
        print(f"Class distribution: {counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, broad_label, severity, current_load = self.samples[idx]

        img_raw = np.asarray(Image.open(img_path), dtype=np.uint16)
        img_norm = normalize_and_resize(img_raw, self.target_size)  # (128, 128) float32 in [0,1]

        img_tensor = torch.from_numpy(img_norm).float().unsqueeze(0)  # (1, 128, 128)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        label_idx = LABEL_TO_IDX[broad_label]
        return img_tensor, label_idx, current_load


if __name__ == "__main__":
    # Quick smoke test — works whether you run this from the project root
    # (python src/thermal_dataset.py) or from inside src/ (python thermal_dataset.py).
    _here = os.path.dirname(os.path.abspath(__file__))          # .../bearing-rca-project/src
    _project_root = os.path.dirname(_here)                       # .../bearing-rca-project
    BASE = os.path.join(_project_root, "dataset", "thermal_squirrel_cage", "thermal_images")

    dataset = ThermalMotorDataset(
        root_dirs=[os.path.join(BASE, "coupling_1"), os.path.join(BASE, "coupling_2")]
    )

    img, label_idx, load = dataset[0]
    print(f"Sample 0 -> image shape: {tuple(img.shape)}, dtype: {img.dtype}, "
          f"label: {IDX_TO_NAME[label_idx]}, current_load: {load}A")