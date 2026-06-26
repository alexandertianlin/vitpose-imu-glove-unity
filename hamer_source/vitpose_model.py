from __future__ import annotations

import os
import sys
import urllib.request

import numpy as np
import torch
import torch.nn as nn

from mmpose.apis import inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result

os.environ["PYOPENGL_PLATFORM"] = "egl"

# project root directory
ROOT_DIR = "./"
VIT_DIR = os.path.join(ROOT_DIR, "third-party/ViTPose")

class ViTPoseModel(object):

    DOWNLOAD_URLS = {
        'ViTPose-base-hand': {
            'url': 'https://github.com/ViTAE-Transformer/ViTPose/releases/download/v1.1.0/vitpose-base-hand.pth',
            'mirror': 'https://hf-mirror.com/ViTAE/ViTPose-base-hand/resolve/main/pytorch_model.bin',
        },
    }

    def _ensure_downloaded(self, name: str, ckpt_path: str) -> None:
        if os.path.exists(ckpt_path):
            return
        if name not in self.DOWNLOAD_URLS:
            print(f'[ERROR] Model weights not found: {ckpt_path}')
            print(f'  Please manually download for {name} and place at:')
            print(f'  {ckpt_path}')
            sys.exit(1)
        urls = self.DOWNLOAD_URLS[name]
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        for src_name, url in urls.items():
            print(f'Downloading {name} from {src_name} ...')
            print(f'  URL: {url}')
            print(f'  -> {ckpt_path}')
            try:
                urllib.request.urlretrieve(url, ckpt_path)
                print(f'  Download complete')
                return
            except Exception as e:
                print(f'  Failed: {e}')
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
        print(f'[ERROR] All download sources failed for {name}')
        print(f'  Please manually download and place at: {ckpt_path}')
        sys.exit(1)

    MODEL_DICT = {
        'ViTPose+-G (multi-task train, COCO)': {
            'config': f'{VIT_DIR}/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py',
            'model': f'{ROOT_DIR}/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth',
        },
        'ViTPose-base-hand': {
            'config': f'{VIT_DIR}/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/hand/ViTPose_base_hand_256x192.py',
            'model': f'{ROOT_DIR}/_DATA/vitpose_ckpts/vitpose-base-hand/hand.pth',
        },
    }

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self.model_name = 'ViTPose+-G (multi-task train, COCO)'
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        ckpt_path = dic['model']
        self._ensure_downloaded(name, ckpt_path)
        model = init_pose_model(dic['config'], ckpt_path, device=self.device)
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: list[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(image, out, kpt_score_threshold,
                                          vis_dot_radius, vis_line_thickness)
        return out, vis

    def predict_pose(
            self,
            image: np.ndarray,
            det_results: list[np.ndarray],
            box_score_threshold: float = 0.5) -> list[dict[str, np.ndarray]]:
        image = image[:, :, ::-1]  # RGB -> BGR
        person_results = process_mmdet_results(det_results, 1)
        out, _ = inference_top_down_pose_model(self.model,
                                               image,
                                               person_results=person_results,
                                               bbox_thr=box_score_threshold,
                                               format='xyxy')
        return out

    def visualize_pose_results(self,
                               image: np.ndarray,
                               pose_results: list[np.ndarray],
                               kpt_score_threshold: float = 0.3,
                               vis_dot_radius: int = 4,
                               vis_line_thickness: int = 1) -> np.ndarray:
        image = image[:, :, ::-1]  # RGB -> BGR
        vis = vis_pose_result(self.model,
                              image,
                              pose_results,
                              kpt_score_thr=kpt_score_threshold,
                              radius=vis_dot_radius,
                              thickness=vis_line_thickness)
        return vis[:, :, ::-1]  # BGR -> RGB
