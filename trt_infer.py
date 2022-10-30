# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import time
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import os
from tqdm import tqdm
from common import (allocate_buffers, do_inference, create_tensorrt_logger)



INPUT_SIZE = 640
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']  # class names

CLASSES_IDS = [i for i in range(len(CLASSES))]

CONF_THRESH = 0.25





def preprocess_ds_nchw(batch_img):
    batch_img_array = np.array([np.array(img) for img in batch_img], dtype=np.float32)
    batch_img_array = batch_img_array / 255.0
    batch_transpose = np.transpose(batch_img_array, (0, 3, 1, 2))

    return batch_transpose


def decode(keep_k, boxes, scores, cls_id):
    results = []
    for idx, k in enumerate(keep_k.reshape(-1)):
        bbox = boxes[idx].reshape((-1, 4))[:k]
        conf = scores[idx].reshape((-1, 1))[:k]
        cid = cls_id[idx].reshape((-1, 1))[:k]
        results.append(np.concatenate((cid, conf, bbox), axis=-1))
    
    return results


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def draw_bbox_cv(orig_img, infer_img, output_img_path, labels, ratio_pad=None, image_id=None, jlist=None):
    bboxes = labels[:, 2:]
    confs = labels[:, 1]
    cids = labels[:, 0]
    bboxes = scale_coords(infer_img.shape[2:], bboxes, orig_img.shape, ratio_pad=ratio_pad).round()
    
    for idx in range(len(labels)):
        
        bbox = bboxes[idx]
        p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        cid = int(cids[idx])
        conf = confs[idx]
        # print("{}: {} {}".format(CLASSES[cid], conf, bbox))
        if jlist is not None:
            if image_id is not None:
                b = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                jlist.append({
                'image_id': image_id,
                'category_id': CLASSES_IDS[cid],
                'bbox': [round(float(x), 3) for x in b],
                'score': round(float(conf), 5)})

        if conf < CONF_THRESH:
            continue

        cv2.rectangle(orig_img, p1, p2, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(orig_img, "{0}: {1:.2f}".format(CLASSES[cid], conf), p1, 0, 0.8, (255, 255, 0), 2)
    
    cv2.imwrite(output_img_path, orig_img)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # new_shape = (height, width)
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def load_images_cv(img_path, new_shape):
    orig_img = cv2.imread(img_path)
    img = letterbox(orig_img.copy(), new_shape, auto=False, scaleup=True)[0]
    img = img[..., [2, 1, 0]] # BGR -> RGB
    images = preprocess_ds_nchw([img])
    
    return images, orig_img




def square_inference(engine, img_root, output_img_root, input_size=640, jlist=None):
    with engine.create_execution_context() as context:
        context.set_binding_shape(0, (1, 3, input_size, input_size))
        new_shape = (input_size, input_size)
        inputs, outputs, bindings, stream = allocate_buffers(engine, context)

        # calculate the speed of preprocess and inference per image
        infer_time, seen =  0.0, 0

        for img_name in sorted(os.listdir(img_root)):
            if os.path.splitext(img_name)[-1] not in ['.jpg', '.png', '.jpeg']:
                continue
            img_path = os.path.join(img_root, img_name)
            if jlist is not None:
                img_id = int(img_name.split(".")[0])
            else:
                img_id = None
            
            images, orig_img = load_images_cv(img_path, new_shape)
            ratio_pad = None
            batch_images = images
            # Hard Coded For explicit_batch and the ONNX model's batch_size = 1
            batch_images = batch_images[np.newaxis, :, :, :]
            outputs_shape, outputs_data, cost = do_inference(batch=batch_images, context=context,
                                                        bindings=bindings, inputs=inputs,
                                                        outputs=outputs, stream=stream)
            

            results = decode(keep_k = outputs_data["BatchedNMS"],
                                boxes = outputs_data["BatchedNMS_1"],
                                scores = outputs_data["BatchedNMS_2"],
                                cls_id = outputs_data["BatchedNMS_3"])
                                
            infer_time += cost
            seen += 1
            # visualize the bbox
            draw_bbox_cv(orig_img, images, os.path.join(output_img_root, img_name),
                         results[0], image_id=img_id, jlist=jlist, ratio_pad=ratio_pad)
        t = (infer_time / seen) * 1E3   # speeds per image
        print(f'Inference: %.2f ms  per image at shape {(1,3, input_size, input_size)}' % t)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Do YOLOV5 inference using TRT')
    parser.add_argument('--input_images_folder', type=str, help='input images path', required=True)
    parser.add_argument('--output_images_folder', type=str, help='output images path', required=True)
    parser.add_argument('--input_size', type=int, default=640, help="Input Size")
    parser.add_argument('--model', type=str, default="yolov5s.engine", help="Model Path")
    

    args = parser.parse_args()

    img_root = args.input_images_folder
    output_img_root = args.output_images_folder
    input_size=args.input_size
    engine_file_path = args.model
    
    if not os.path.exists(output_img_root):
        print("Please create the output images directory: {output_img_root}")
        exit(0)
    
    trt_logger = create_tensorrt_logger(verbose=True)

    trt.init_libnvinfer_plugins(None, '')
    with open(engine_file_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine
        square_inference(engine, img_root, output_img_root, input_size=input_size)

