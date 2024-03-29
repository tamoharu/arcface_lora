from typing import List
from config.type import DetectFaceModel, SwapFaceModel, MaskFaceModel, MaskFaceRegion, EnhanceFaceModel, Process, TempFrameFormat, OutputVideoEncoder, OutputVideoPreset, LogLevel


thread: int = 32
queue: int = 4
execution_providers = ['CPUExecutionProvider']
source_paths : List[str] = None
target_path : str = None
output_path : str = None
score_threshold: float = 0.35
iou_threshold: float = 0.4
detect_face_model: DetectFaceModel  = 'yolov8'
# mask_face_model: MaskFaceModel = 'face_occluder'
mask_face_model: MaskFaceModel = 'box'
mask_face_regions: List[MaskFaceRegion] = ['right-eye', 'left-eye']
# enhance_face_model: EnhanceFaceModel = 'codeformer'
enhance_face_model: EnhanceFaceModel = None
swap_face_model: SwapFaceModel  = 'inswapper'
process_mode: Process = 'swap'
target_file_types = ['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.mov']

trim_frame_start : int = None
trim_frame_end : int = None
temp_frame_format : TempFrameFormat = 'jpg'
temp_frame_quality : int = 100
keep_temp : bool = False
keep_fps : bool = False

output_image_quality : int = 100
output_video_encoder : OutputVideoEncoder = 'libx264'
output_video_preset : OutputVideoPreset = 'veryfast'
output_video_quality : int = 100
output_video_fps : float = 25

log_level : LogLevel = 'info'

webcam = False

model_type = 'torch'
model_name = 'arcface_w600k_r50.pth'