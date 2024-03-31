import sys
import os
sys.path.append('..')
from swap_face.core import main
from utils.vision import read_static_images, read_static_image, write_image
import config.globals as globals


def test_swap():
    # globals.model_type = 'torch'
    globals.model_type = 'onnx'
    # globals.model_name = 'trained_model.pth'
    # globals.model_name = 'arcface_w600k_r50.pth'
    globals.model_name = 'w600k_mbf.onnx'
    sources_path = ['./assets/hou/hou1.jpg', './assets/hou/hou2.jpg']
    target_path = './assets/hashikan/hashikan5.jpg'
    sources = read_static_images(sources_path)
    target = read_static_image(target_path)
    result = main(sources, target)

    output_dir = './outputs/'
    base_filename = 'result'
    extension = '.jpg'
    counter = 1

    while True:
        output_filename = f"{base_filename}{counter}{extension}"
        output_path = os.path.join(output_dir, output_filename)
        if not os.path.exists(output_path):
            break
        counter += 1

    write_image(output_path, result)

if __name__ == '__main__':
    test_swap()