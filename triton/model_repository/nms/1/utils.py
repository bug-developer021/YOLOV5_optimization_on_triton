
import numpy as np
import torch
import torchvision
from torch.nn import functional as F
import time

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def postprocess(output, conf_th=0.25, nms_threshold=0.45, max_det=300):
    """Postprocess TensorRT outputs.
    # Args
        output: list of detections with schema 
        [batch_size, num_detections, xywh + conf + cls_id]

        conf_th: confidence threshold
        nms_threshold: nms threshold

    # Returns
        list of bounding boxes with all detections above threshold and after nms, see class BoundingBox
        [num_detections , xyxy + conf + cls_id] * batch_size
    """
    
    # Get the num of boxes detected
    output_candidates = output[..., 4] > conf_th
    output_bboxes = [torch.zeros((0, 6), device=output.device)] * output.shape[0]
    for xi, x in enumerate(output):
        # Apply confidence constraints
        x = x[output_candidates[xi]]

        if not x.shape[0]:
            continue

        boxes = xywh2xyxy(x[:, :4])
        scores = x[:, 4]
        i = torchvision.ops.nms(boxes, scores, nms_threshold)

        # padding boxes to 300
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        
        bbox_pad_nums = max_det - i.shape[0]
        
        output_bboxes[xi] = F.pad(x[i], (0,0,0, bbox_pad_nums), value=0)
        # output_bboxes[xi] = x[i]
    
    return torch.stack(output_bboxes, dim=0)

def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()