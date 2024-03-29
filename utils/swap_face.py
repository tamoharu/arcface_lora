from typing import List, Tuple
import threading
import os
import shutil
import time

import cv2
import numpy as np

import config.instances as instances
import config.face_store as face_store
import config.globals as globals
import utils.logger as logger
from config.type import Frame, Kps, Matrix, Embedding, Template, Size, MaskFaceModel, Mask
from utils.model_zoo.yolov8 import Yolov8
from utils.model_zoo.arcface_torch import ArcfaceTorch
from utils.model_zoo.arcface_onnx import ArcfaceOnnx
from utils.model_zoo.inswapper import Inswapper
from utils.vision import read_static_images, read_static_image, write_image, is_image
from utils.filesystem import clear_temp
from utils.ffmpeg import compress_image


def resolve_relative_path(path : str) -> str:
	return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def detect_face(frame):
    model = Yolov8(model_path="../models/yolov8n-face.onnx", execution_providers=["CPUExecutionProvider"], score_threshold=0.35, iou_threshold=0.4)
    return model.predict(frame)


def mask_face(frame: Frame, model_name: MaskFaceModel) -> Mask:
    crop_size = frame.shape[:2][::-1]
    mask_padding = (0, 0, 0, 0)
    mask_blur = 0.3
    blur_amount = int(crop_size[0] * 0.5 * mask_blur)
    blur_area = max(blur_amount // 2, 1)
    box_mask = np.ones(crop_size, np.float32)
    box_mask[:max(blur_area, int(crop_size[1] * mask_padding[0] / 100)), :] = 0
    box_mask[-max(blur_area, int(crop_size[1] * mask_padding[2] / 100)):, :] = 0
    box_mask[:, :max(blur_area, int(crop_size[0] * mask_padding[3] / 100))] = 0
    box_mask[:, -max(blur_area, int(crop_size[0] * mask_padding[1] / 100)):] = 0
    box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
    return box_mask


def embed_face(frame, kps):
    if globals.model_type == 'torch':
        model = ArcfaceTorch(model_path=resolve_relative_path('../models/' + globals.model_name))
    elif globals.model_type == 'onnx':
        model = ArcfaceOnnx(model_path=resolve_relative_path('../models/' + globals.model_name), execution_providers=globals.execution_providers)
    else:
        raise NotImplementedError(f"Model {globals.model_type} not implemented.")
    return model.predict(frame, kps) 


def model_router():
    if globals.swap_face_model == 'inswapper':
        return Inswapper(
            model_path=resolve_relative_path('../models/inswapper_128.onnx'),
            execution_providers=globals.execution_providers
        )
    else:
        raise NotImplementedError(f"Model {globals.swap_face_model} not implemented.")


def create_source_embedding(source_frames: List[Frame]):
    with threading.Lock():
        if face_store.source_embedding is None:
            embedding_list = []
            for source_frame in source_frames:
                _, source_kps_list, _ = detect_face(frame=source_frame)
                source_kps = source_kps_list[0]
                embedding = embed_face(frame=source_frame, kps=source_kps)
                embedding_list.append(embedding)
            face_store.source_embedding = np.mean(embedding_list, axis=0)


def swap_face(source_embedding: Embedding, target_frame: Frame) -> Frame:
    _, target_kps_list, _ = detect_face(frame=target_frame)
    temp_frame = target_frame
    for target_kps in target_kps_list:
        temp_frame = apply_swap(temp_frame, target_frame, target_kps, source_embedding)
    return temp_frame


def apply_swap(temp_frame, target_frame: Frame, target_kps: Kps, embedding: Embedding) -> Frame:
    model = model_router()
    crop_frame, affine_matrix = warp_face_kps(target_frame, target_kps, model.model_template, model.model_size)
    crop_mask = mask_face(frame=crop_frame, model_name=globals.mask_face_model)
    crop_frame = model.predict(target_crop_frame=crop_frame, source_embedding=embedding)
    result_frame = paste_back(temp_frame, crop_frame, crop_mask, affine_matrix)
    return result_frame


def warp_face_kps(temp_frame: Frame, kps: Kps, model_template: Template, model_size: Size) -> Tuple[Frame, Matrix]:
    normed_template = model_template * model_size
    affine_matrix = cv2.estimateAffinePartial2D(kps, normed_template, method = cv2.RANSAC, ransacReprojThreshold = 100)[0]
    crop_frame = cv2.warpAffine(temp_frame, affine_matrix, model_size, borderMode = cv2.BORDER_REPLICATE, flags = cv2.INTER_AREA)
    return crop_frame, affine_matrix


def paste_back(target_frame: Frame, crop_frame: Frame, crop_mask: Frame, affine_matrix: Matrix) -> Frame:
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)
    temp_frame_size = target_frame.shape[:2][::-1]
    inverse_crop_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_frame_size).clip(0, 1)
    inverse_crop_frame = cv2.warpAffine(crop_frame, inverse_matrix, temp_frame_size, borderMode = cv2.BORDER_REPLICATE)
    paste_frame = target_frame.copy()
    paste_frame[:, :, 0] = inverse_crop_mask * inverse_crop_frame[:, :, 0] + (1 - inverse_crop_mask) * target_frame[:, :, 0]
    paste_frame[:, :, 1] = inverse_crop_mask * inverse_crop_frame[:, :, 1] + (1 - inverse_crop_mask) * target_frame[:, :, 1]
    paste_frame[:, :, 2] = inverse_crop_mask * inverse_crop_frame[:, :, 2] + (1 - inverse_crop_mask) * target_frame[:, :, 2]
    return paste_frame


def clear() -> None:
	instances.clear_instances()
	face_store.reset_face_store()


def process_image(start_time : float) -> None:
    clear_temp(globals.target_path)
    clear()
    shutil.copy2(globals.target_path, globals.output_path)
    if globals.process_mode == 'swap':
        source_frames = read_static_images(globals.source_paths)
        target_frame = read_static_image(globals.target_path)
        create_source_embedding(source_frames)
        source_embedding = face_store.source_embedding
        result_frame = swap_face(source_embedding ,target_frame)
        write_image(globals.output_path, result_frame)
    logger.info('compressing_image', __name__.upper())
    if not compress_image(globals.output_path):
        logger.error('compressing_image_failed', __name__.upper())
    if is_image(globals.output_path):
        seconds = '{:.2f}'.format((time.time() - start_time) % 60)
        logger.info(f'processing_image_succeed: {seconds} seconds', __name__.upper())
    else:
        logger.error('processing_image_failed', __name__.upper())