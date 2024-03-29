from typing import List
import threading

import numpy as np
import cv2

import config.instances as instances
from config.type import Frame, Kps, Embedding, Output
from utils.model_zoo._base_model import OnnxBaseModel


class ArcfaceOnnx(OnnxBaseModel):
    '''
    input
    input.1: ['None', 3, 112, 112]

    output
    683: [1, 512]
    '''
    _lock = threading.Lock()
    def __init__(self, model_path: str, execution_providers: List[str]):
        with ArcfaceOnnx._lock:
            if instances.arcface_inswapper_instance is None:
                super().__init__(model_path, execution_providers)
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
        output = self.forward(crop_frame)
        embedding = self.post_process(output)
        return embedding
    

    def pre_process(self, frame: Frame, kps: Kps) -> Frame:
        crop_frame = self.warp_face_kps(frame, kps)
        crop_frame = crop_frame.astype(np.float32) / 127.5 - 1
        crop_frame = crop_frame[:, :, ::-1].transpose(2, 0, 1)
        crop_frame = np.expand_dims(crop_frame, axis = 0)
        return crop_frame


    def forward(self, frame: Frame) -> Output:
        with self.semaphore:
            output = self.session.run(None,
            {
                self.input_names[0]: frame,
            })
        return output
    

    def post_process(self, output: Output) -> Embedding:
        return output[0].ravel()

    
    def warp_face_kps(self, frame, kps: Kps) -> Frame:
        normed_template = np.array(self.model_template) * np.array(self.model_size)
        self.affine_matrix = cv2.estimateAffinePartial2D(np.array(kps), normed_template, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
        crop_frame = cv2.warpAffine(frame, self.affine_matrix, self.model_size, borderMode = cv2.BORDER_REPLICATE, flags = cv2.INTER_AREA)
        return crop_frame