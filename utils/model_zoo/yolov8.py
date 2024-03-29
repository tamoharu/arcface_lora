from typing import List, Tuple
import threading

import cv2
import numpy as np

import config.instances as instances
from config.type import Frame, Bbox, Kps, Score, Output
from utils.model_zoo._base_model import OnnxBaseModel


class Yolov8(OnnxBaseModel):
    '''
    input
    images: [n, 3, 640, 640]

    output
    output0: [n, 20, 8400]
    '''
    _lock = threading.Lock()
    def __init__(self, model_path: str, execution_providers: List[str], score_threshold: float, iou_threshold: float):
        with Yolov8._lock:
            if instances.yolov8_instance is None:
                super().__init__(model_path, execution_providers)
                self.model_size = (640, 640)
                self.score_threshold = score_threshold
                self.iou_threshold = iou_threshold
                instances.yolov8_instance = self
            else:
                self.__dict__ = instances.yolov8_instance.__dict__

    
    def predict(self, frame) -> Tuple[List[Bbox], List[Kps], List[Score]]:
        frame, resize_data = self.pre_process(frame)
        output = self.forward(frame)
        bbox_list, kps_list, score_list = self.post_process(output, resize_data)
        return bbox_list, kps_list, score_list


    def pre_process(self, frame) -> Tuple[Frame, Tuple[int, int, float]]:
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


    def forward(self, frame) -> Output:
        with self.semaphore:
            output = self.session.run(None,
            {
                self.input_names[0]: frame
            })
        return output

    
    def post_process(self, detections: Output, resize_data) -> Tuple[List[Bbox], List[Kps], List[Score]]:
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
    

    def apply_nms(self, bbox_list, iou_threshold : float) -> List[int]:
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