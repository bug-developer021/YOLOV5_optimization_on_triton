# generate real input data for triton perf_analyzer

import sys

sys.path.append('../')
from trt_infer import load_images_cv
import argparse
import numpy as np
import json
import os
from tqdm import tqdm





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_images', type=str, help='input images', required=True)
    parser.add_argument('--output_file', type=str, help='output file', required=True)
    parser.add_argument('--input_size', type=int, default=640, help="Input Size")
    parser.add_argument('--input_tensor_name', type=str, default='images', help="Input Tensor Name")
    

    args = parser.parse_args()

    img_root = args.input_images
    output_file = args.output_file
    input_size=args.input_size
    input_tensor_name = args.input_tensor_name
    new_shape = (input_size, input_size)

    triton_input_list = []
    for img_name in tqdm(sorted(os.listdir(img_root))):
        if os.path.splitext(img_name)[-1] not in ['.jpg', '.png', '.jpeg']:
            continue
        img_path = os.path.join(img_root, img_name)

        input_image, _ = load_images_cv(img_path, new_shape)
        input_image = np.squeeze(input_image)
        triton_input_shape = input_image.shape

        # flatten_img = input_image.flatten().astype(np.float16)
        flatten_img = input_image.flatten()
        
        triton_input = {
            input_tensor_name:
                {
                    "content": flatten_img.tolist(),
                    "shape": list(triton_input_shape)
                }
        }
        triton_input_list.append(triton_input)

    with open(output_file, "w") as f:
        json.dump(
            {"data": triton_input_list}, f
        )
    

