from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import yaml
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from iglovikov_helper_functions.dl.pytorch.utils import state_dict_from_disk, tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad_to_size, unpad_from_size
import onnx
from onnxsim import simplify
from sor4onnx import rename
from snc4onnx import combine

class Pre_post_model(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
    ):
        super(Pre_post_model, self).__init__()
        self.model = model
        self.mean = torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32)).reshape(1,3,1,1)
        self.std = torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32)).reshape(1,3,1,1)

    def forward(self, x: torch.Tensor):
        x = (x - self.mean) / self.std
        x = self.model(x)
        return x

def main():
    with open("./configs/2020-09-23a.yaml") as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    model: torch.nn.Module = object_from_dict(hparams["model"])
    corrections: Dict[str, str] = {"model.": ""}
    state_dict = state_dict_from_disk(file_path="./2020-09-23a.pth", rename_in_layers=corrections)
    model.load_state_dict(state_dict)
    model.eval()
    model.cpu()

    pre_post_model = Pre_post_model(model)

    RESOLUTION = [
        # [192,320],
        # [192,416],
        # [192,640],
        # [192,800],
        # [256,320],
        # [256,416],
        # [256,448],
        # [256,640],
        # [256,800],
        # [256,960],
        # [288,480],
        # [288,640],
        # [288,800],
        # [288,960],
        # [288,1280],
        # [320,320],
        # [384,480],
        # [384,640],
        # [384,800],
        # [384,960],
        # [384,1280],
        # [416,416],
        [480,640],
        # [480,800],
        # [480,960],
        # [480,1280],
        # [512,512],
        # [512,640],
        # [512,896],
        # [544,800],
        # [544,960],
        # [544,1280],
        # [640,640],
        # [736,1280],
    ]

    for H, W in RESOLUTION:
        onnx_file = f"peopleseg_1x3x{H}x{W}.onnx"
        x = torch.randn(1, 3, H, W).cpu()
        torch.onnx.export(
            pre_post_model,
            args=(x),
            f=onnx_file,
            opset_version=13,
            input_names=['input_rgb'],
            output_names=['segment'],
        )
        model_onnx1 = onnx.load(onnx_file)
        model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
        onnx.save(model_onnx1, onnx_file)

        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)

        onnx_graph = rename(
            old_new=["/", "seg/"],
            input_onnx_file_path=onnx_file,
            output_onnx_file_path=onnx_file,
            mode="full",
            search_mode="prefix_match",
        )

        combined_graph = combine(
            srcop_destop = [
                ['output_prep', 'input_rgb']
            ],
            input_onnx_file_paths = [
                f'yolov9_e_wholebody34_post_0100_1x3x{H}x{W}.onnx',
                f'peopleseg_1x3x{H}x{W}.onnx',
            ],
            output_onnx_file_path = f'yolov9_e_wholebody34_with_seg_post_0100_1x3x{H}x{W}.onnx',
        )

if __name__ == "__main__":
    main()
