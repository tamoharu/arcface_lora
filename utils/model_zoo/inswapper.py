from typing import List, Tuple
import threading

import numpy as np
import onnx

import config.instances as instances
from config.type import Frame, Embedding, Output, Matrix, Size, Template
from utils.model_zoo._base_model import OnnxBaseModel


class Inswapper(OnnxBaseModel):
    '''
    input
    target: [1, 3, 128, 128]
    source: [1, 512]

    output
    output: [1, 3, 128, 128]
    '''
    _lock = threading.Lock()
    def __init__(self, model_path: str, execution_providers: List[str]):
        with Inswapper._lock:
            if instances.inswapper_instance is None:
                super().__init__(model_path, execution_providers)
                self.model_size = (128, 128)
                self.model_template = np.array(
                [
                    [ 0.36167656, 0.40387734 ],
                    [ 0.63696719, 0.40235469 ],
                    [ 0.50019687, 0.56044219 ],
                    [ 0.38710391, 0.72160547 ],
                    [ 0.61507734, 0.72034453 ]
                ])
                self.model_matrix: Matrix = None
                with self.lock:
                    if self.model_matrix is None:
                        model = onnx.load(model_path)
                        self.model_matrix = onnx.numpy_helper.to_array(model.graph.initializer[-1])
                instances.inswapper_instance = self
            else:
                self.__dict__ = instances.inswapper_instance.__dict__


    def predict(self, target_crop_frame: Frame, source_embedding: Embedding) -> Frame:
        source_embedding = self.prepare_source(source_embedding)
        target_frame = self.prepare_target(target_crop_frame)
        output = self.forward(target_frame, source_embedding)
        crop_frame = self.post_process(output)
        return crop_frame


    def forward(self, target_frame: Frame, source_embedding: Embedding) -> Output:
        with self.semaphore:
            output = self.session.run(None,
            {
                self.input_names[0]: target_frame,
                self.input_names[1]: source_embedding
            })
        return output


    def post_process(self, output: Output) -> Frame:
        crop_frame = output[0][0]
        crop_frame = (crop_frame.astype(np.float32) * 255.0).round()
        crop_frame = crop_frame.transpose(1, 2, 0)
        crop_frame = crop_frame[:, :, ::-1]
        return crop_frame


    def prepare_source(self, embedding: Embedding) -> Embedding:
        embedding = embedding.reshape((1, -1))
        embedding = np.dot(embedding, self.model_matrix) / np.linalg.norm(embedding)
        return embedding


    def prepare_target(self, crop_frame: Frame) -> Frame:
        crop_frame = crop_frame[:, :, ::-1] / np.array([255.0])
        crop_frame = (crop_frame) / [1.0, 1.0, 1.0]
        crop_frame = crop_frame.transpose(2, 0, 1)
        crop_frame = np.expand_dims(crop_frame, axis = 0).astype(np.float32)
        return crop_frame


    def get_model_info(self) -> Tuple[Template, Size]:
        return self.model_template, self.model_size