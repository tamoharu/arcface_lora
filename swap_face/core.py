import os

os.environ['OMP_NUM_THREADS'] = '1'
import time
import tempfile

import cv2

import config.globals as globals
from utils.swap_face import process_image
from utils.vision import read_static_image


def save_file(file_path: str, frame):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(file_path, frame)

def main(sources, target):
    source_paths = []
    for i, source in enumerate(sources):
        source_path = os.path.join(tempfile.mkdtemp(), os.path.basename(f'source{i}.jpg'))
        save_file(source_path, source)
        source_paths.append(source_path)
    target_path = os.path.join(tempfile.mkdtemp(), os.path.basename(f'target.jpg'))
    save_file(target_path, target)
    globals.source_paths = source_paths
    globals.target_path = target_path
    globals.output_path = os.path.join(tempfile.mkdtemp(), os.path.basename(f'output.jpg'))
    process_image(time.time())
    output = read_static_image(globals.output_path)
    return output