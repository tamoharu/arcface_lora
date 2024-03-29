import sys

import onnx
import torch

sys.path.append("../")
from onnx2pytorch import ConvertModel


onnx_model = onnx.load("./models/arcface_w600k_r50.onnx")
pytorch_model = ConvertModel(onnx_model, experimental=True)
torch.save(pytorch_model, "./models/arcface_w600k_r50.pth")