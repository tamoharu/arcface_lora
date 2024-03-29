from threading import Lock, Semaphore

import torch
import numpy as np
import cv2

import config.instances as instances
from config.type import Frame, Kps, Embedding


class TorchBaseModel:
    _instances = {}

    @classmethod
    def get_instance(cls, model_path: str):
        if cls not in cls._instances:
            cls._instances[cls] = cls(model_path)
        return cls._instances[cls]
    
    def __init__(self, model_path: str):
        self.lock: Lock = Lock()
        self.semaphore: Semaphore = Semaphore()
        self.model_path = model_path
        self.model = self.load_model(self.model_path)

    def load_model(self, model_path):
        model = torch.load(model_path)
        model.eval()
        return model


class ArcfaceTorch(TorchBaseModel):
    _lock = Lock()
    
    def __init__(self, model_path: str):
        with ArcfaceTorch._lock:
            if instances.arcface_inswapper_instance is None:
                super().__init__(model_path)
                self.model_size = (112, 112)
                self.model_template = np.array(
                [
                    [ 0.36167656, 0.40387734 ],
                    [ 0.63696719, 0.40235469 ],
                    [ 0.50019687, 0.56044219 ],
                    [ 0.38710391, 0.72160547 ],
                    [ 0.61507734, 0.72034453 ]
                ])
                instances.arcface_inswapper_instance = self
            else:
                self.__dict__ = instances.arcface_inswapper_instance.__dict__
    
    def predict(self, frame: Frame, kps: Kps) -> Embedding:
        crop_frame = self.pre_process(frame, kps)
        crop_frame = crop_frame.copy()
        crop_frame_tensor = torch.from_numpy(crop_frame).float()
        output = self.forward(crop_frame_tensor)
        embedding = self.post_process(output)
        return embedding

    def pre_process(self, frame: Frame, kps: Kps) -> np.ndarray:
        crop_frame = self.warp_face_kps(frame, kps)
        crop_frame = crop_frame.astype(np.float32) / 127.5 - 1
        crop_frame = crop_frame[:, :, ::-1].transpose(2, 0, 1)
        crop_frame = np.expand_dims(crop_frame, axis=0)
        return crop_frame

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        frame = torch.tensor(frame, dtype=torch.float32)
        if torch.cuda.is_available():
            frame = frame.to('cuda')
            self.model.to('cuda')
        with torch.no_grad():
            output = self.model(frame)
        return output

    def post_process(self, output: torch.Tensor) -> np.ndarray:
        return output.cpu().numpy().ravel()

    def warp_face_kps(self, frame: np.ndarray, kps: Kps) -> np.ndarray:
        normed_template = self.model_template * np.array(self.model_size)
        affine_matrix = cv2.estimateAffinePartial2D(np.array(kps), normed_template, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
        crop_frame = cv2.warpAffine(frame, affine_matrix, self.model_size, borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
        return crop_frame