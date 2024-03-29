from typing import List
from threading import Lock, Semaphore

import numpy as np
import cv2
import onnxruntime
import torch


model_type = "onnx"


class OnnxBaseModel:
    _instances = {}

    @classmethod
    def get_instance(cls, model_path: str, execution_providers: List[str]):
        if cls not in cls._instances:
            cls._instances[cls] = cls(model_path, execution_providers)
        return cls._instances[cls]
    
    
    def __init__(self, model_path: str, execution_providers: List[str]):
        self.lock: Lock = Lock()
        self.semaphore: Semaphore = Semaphore()
        self.model_path = model_path
        self.execution_providers = execution_providers
        self.session: onnxruntime.InferenceSession = None
        with self.lock:
            if model_type == "onnx":
                if self.session is None:
                    self.session = onnxruntime.InferenceSession(self.model_path, providers = self.execution_providers)
                inputs = self.session.get_inputs()
                self.input_names = []
                for input in inputs:
                    self.input_names.append(input.name)
                outputs = self.session.get_outputs()
                self.output_names = []
                for output in outputs:
                    self.output_names.append(output.name)
                onnxruntime.set_default_logger_severity(3)


class ArcfaceInswapper(OnnxBaseModel):
    '''
    input
    input.1: ['None', 3, 112, 112]

    output
    683: [1, 512]
    '''
    _lock = Lock()
    def __init__(self, model_path: str, execution_providers: List[str]):
        with ArcfaceInswapper._lock:
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
            if model_type == "torch":
                self.model = torch.load(self.model_path)
                self.model.eval()
                self.model.to(self.execution_providers[0])
    

    def predict(self, frame, kps):
        crop_frame = self.pre_process(frame, kps)
        output = self.forward(crop_frame)
        embedding = self.post_process(output)
        return embedding
    

    def pre_process(self, frame, kps):
        crop_frame = self.warp_face_kps(frame, kps)
        crop_frame = crop_frame.astype(np.float32) / 127.5 - 1
        crop_frame = crop_frame[:, :, ::-1].transpose(2, 0, 1)
        crop_frame = np.expand_dims(crop_frame, axis = 0)
        return crop_frame


    def forward(self, frame):
        if model_type == "onnx":
            with self.semaphore:
                output = self.session.run(None,
                {
                    self.input_names[0]: frame,
                })
            return output
        elif model_type == "torch":
            frame = torch.from_numpy(frame).float()
            frame = frame.to(self.execution_providers[0])
            with torch.no_grad():
                output = self.model(frame)
            return output.cpu().numpy()


    def post_process(self, output):
        return output[0].ravel()

    
    def warp_face_kps(self, frame, kps):
        normed_template = np.array(self.model_template) * np.array(self.model_size)
        self.affine_matrix = cv2.estimateAffinePartial2D(np.array(kps), normed_template, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
        crop_frame = cv2.warpAffine(frame, self.affine_matrix, self.model_size, borderMode = cv2.BORDER_REPLICATE, flags = cv2.INTER_AREA)
        return crop_frame
    

class Yolov8():
    '''
    input
    images: [n, 3, 640, 640]

    output
    output0: [n, 20, 8400]
    '''
    def __init__(self, model_path: str, execution_providers: List[str], score_threshold: float, iou_threshold: float):
        self.lock: Lock = Lock()
        self.semaphore: Semaphore = Semaphore()
        self.model_path = model_path
        self.execution_providers = execution_providers
        self.session: onnxruntime.InferenceSession = None
        with self.lock:
            if self.session is None:
                self.session = onnxruntime.InferenceSession(self.model_path, providers = self.execution_providers)
        inputs = self.session.get_inputs()
        self.input_names = []
        for input in inputs:
            self.input_names.append(input.name)
        outputs = self.session.get_outputs()
        self.output_names = []
        for output in outputs:
            self.output_names.append(output.name)
        onnxruntime.set_default_logger_severity(3)
        self.model_size = (640, 640)
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    
    def predict(self, frame):
        frame, resize_data = self.pre_process(frame)
        output = self.forward(frame)
        bbox_list, kps_list, score_list = self.post_process(output, resize_data)
        return bbox_list, kps_list, score_list


    def pre_process(self, frame):
        frame_height, frame_width = frame.shape[:2]
        resize_ratio = min(self.model_size[0] / frame_height, self.model_size[1] / frame_width)
        resized_shape = int(round(frame_width * resize_ratio)), int(round(frame_height * resize_ratio))
        frame = cv2.resize(frame, resized_shape, interpolation=cv2.INTER_LINEAR)
        offset_height = (self.model_size[0] - resized_shape[1]) / 2
        offset_width = (self.model_size[1] - resized_shape[0]) / 2
        resize_data = tuple([offset_height, offset_width, resize_ratio])
        frame = cv2.copyMakeBorder(frame, round(offset_height - 0.1), round(offset_height + 0.1), round(offset_width - 0.1), round(offset_width + 0.1), cv2.BORDER_CONSTANT, value = (114, 114, 114))
        frame = frame.astype(np.float32) / 255.0
        frame = frame[..., ::-1].transpose(2, 0, 1)
        frame = np.expand_dims(frame, axis = 0)
        frame = np.ascontiguousarray(frame)
        return frame, resize_data


    def forward(self, frame):
        with self.semaphore:
            output = self.session.run(None,
            {
                self.input_names[0]: frame
            })
        return output

    
    def post_process(self, detections, resize_data):
        bbox_list = []
        kps_list = []
        score_list = []
        detections = np.squeeze(detections).T
        bbox_raw, score_raw, kps_raw = np.split(detections, [ 4, 5 ], axis = 1)
        keep_indices = np.where(score_raw > self.score_threshold)[0]
        if keep_indices.any():
            bbox_raw, kps_raw, score_raw = bbox_raw[keep_indices], kps_raw[keep_indices], score_raw[keep_indices]
            offset_height, offset_width, resize_ratio = resize_data
            for bbox in bbox_raw:
                bbox_list.append(np.array(
                [
                    (bbox[0] - bbox[2] / 2 - offset_width) / resize_ratio,
                    (bbox[1] - bbox[3] / 2 - offset_height) / resize_ratio,
                    (bbox[0] + bbox[2] / 2 - offset_width) / resize_ratio,
                    (bbox[1] + bbox[3] / 2 - offset_height) / resize_ratio
                ]))
            kps_raw[:, 0::3] = (kps_raw[:, 0::3] - offset_width) / resize_ratio
            kps_raw[:, 1::3] = (kps_raw[:, 1::3] - offset_height) / resize_ratio
            for kps in kps_raw:
                indexes = np.arange(0, len(kps), 3)
                temp_kps = []
                for index in indexes:
                    temp_kps.append([kps[index], kps[index + 1]])
                kps_list.append(np.array(temp_kps))
            score_list = score_raw.ravel().tolist()
            nms_keep_indices = self.apply_nms(bbox_list, self.iou_threshold)
            bbox_list = [bbox_list[i] for i in nms_keep_indices]
            kps_list = [kps_list[i] for i in nms_keep_indices]
            score_list = [score_list[i] for i in nms_keep_indices]
        return bbox_list, kps_list, score_list
    

    def apply_nms(self, bbox_list, iou_threshold):
        keep_indices = []
        dimension_list = np.reshape(bbox_list, (-1, 4))
        x1 = dimension_list[:, 0]
        y1 = dimension_list[:, 1]
        x2 = dimension_list[:, 2]
        y2 = dimension_list[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        indices = np.arange(len(bbox_list))
        while indices.size > 0:
            index = indices[0]
            remain_indices = indices[1:]
            keep_indices.append(index)
            xx1 = np.maximum(x1[index], x1[remain_indices])
            yy1 = np.maximum(y1[index], y1[remain_indices])
            xx2 = np.minimum(x2[index], x2[remain_indices])
            yy2 = np.minimum(y2[index], y2[remain_indices])
            width = np.maximum(0, xx2 - xx1 + 1)
            height = np.maximum(0, yy2 - yy1 + 1)
            iou = width * height / (areas[index] + areas[remain_indices] - width * height)
            indices = indices[np.where(iou <= iou_threshold)[0] + 1]
        return keep_indices


def detect_face(frame):
    model = Yolov8(model_path="../models/yolov8n-face.onnx", execution_providers=["CPUExecutionProvider"], score_threshold=0.35, iou_threshold=0.4)
    return model.predict(frame)


def process_frame(frame):
    _, kps, _ = detect_face(frame)
    model = ArcfaceInswapper(model_path="../models/model.onnx", execution_providers=["cpu"])
    return model.predict(frame, kps)


def test_onnx_model():
    frame = cv2.imread("../data/class1/IMG_0421.JPG")
    embedding = process_frame(frame)
    print(embedding)


if __name__ == "__main__":
    test_onnx_model()
