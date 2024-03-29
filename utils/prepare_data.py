import numpy as np
import cv2
import os
import glob
import shutil

from utils.model_zoo.yolov8 import Yolov8


def process_all_images(input_dir, output_dir):
    extensions = ['*.jpg', '*.JPG', '*.png', '*.PNG']
    for extension in extensions:
        for img_path in glob.glob(os.path.join(input_dir, '**', extension), recursive=True):
            frame = cv2.imread(img_path)
            crop_frame = process_frame(frame)
            relative_path = os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, crop_frame)


def process_frame(frame):
    model_size = (112, 112)
    model_template = np.array(
    [
        [ 0.36167656, 0.40387734 ],
        [ 0.63696719, 0.40235469 ],
        [ 0.50019687, 0.56044219 ],
        [ 0.38710391, 0.72160547 ],
        [ 0.61507734, 0.72034453 ]
    ])

    _, kps, _ = detect_face(frame)
    normed_template = np.array(model_template) * np.array(model_size)
    affine_matrix = cv2.estimateAffinePartial2D(np.array(kps), normed_template, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
    crop_frame = cv2.warpAffine(frame, affine_matrix, model_size, borderMode = cv2.BORDER_REPLICATE, flags = cv2.INTER_AREA)
    return crop_frame


def detect_face(frame):
    model = Yolov8(model_path="./models/yolov8n-face.onnx", execution_providers=["CPUExecutionProvider"], score_threshold=0.35, iou_threshold=0.4)
    return model.predict(frame)