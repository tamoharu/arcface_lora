from typing import Any, Literal, Tuple, Callable, List

import numpy as np


Frame = np.ndarray[Any, Any]
Bbox = np.ndarray[Any, Any]
Kps = np.ndarray[Any, Any]
Score = float
Embedding = np.ndarray[Any, Any]
Output = np.ndarray[Any, Any]
Mask = np.ndarray[Any, Any]
Matrix = np.ndarray[Any, Any]
Template = np.ndarray[Any, Any]
Size = Tuple[int, int]
UpdateProcess = Callable[[], None]
Resolution = Tuple[int, int]
ProcessFrames = Callable[[List[str], List[str], UpdateProcess], None]
Process = Literal['swap', 'blur']
DetectFaceModel = Literal['yolov8']
MaskFaceModel = Literal['face_occluder', 'face_parser', 'box']
EnhanceFaceModel = Literal['codeformer']
SwapFaceModel = Literal['inswapper']
MaskFaceRegion = Literal['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'eye-glasses', 'nose', 'mouth', 'upper-lip', 'lower-lip']
TempFrameFormat = Literal['jpg', 'png', 'bmp']
OutputVideoEncoder = Literal['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc']
OutputVideoPreset = Literal['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']
ProcessMode = Literal['output', 'preview', 'stream']
LogLevel = Literal['error', 'warn', 'info', 'debug']
VideoMemoryStrategy = Literal['strict', 'moderate', 'tolerant']
Fps = float