# YOLOV5 TensorRT BatchedNMS

In [Modifying the yolov5 detect layer](custom_yolov5_detect_layer_EN.md), the lightweight optimization of the detect layer is introduced to improve model serving efficiency. In [Deploying yolov5 Triton Pipelines](pipelines_EN.md), Triton Pipelines are deployed through BLS and Ensemble respectively. However, the infer engine and NMS in Pipelines are two relatively independent steps, where NMS is completed through the python backend. Both BLS and Ensemble have some limitations in data transfer.

This article utilizes onnx_graphsurgeon to modify the output tensor of the original detect layer, connects it to the TensorRT batchedNMSPlugin implemented by cuda, and integrates yolov5 NMS into the tensorrt engine, avoiding device to host data copies in some scenarios and improving overall computational performance.

## 0. Prerequisites

```shell
# clone ultralytics repo
git clone -b v6.1 https://github.com/ultralytics/yolov5.git
# clone this repo
git clone <this repo>  
cp -r <this repo>/* yolov5/
```

---

## 1. Specific Steps 

It is similar to [Modifying the yolov5 detect layer](custom_yolov5_detect_layer_EN.md#3-specific-steps), following the steps of:

- Modify the detect layer's forward function
- Export the .onnx file
- Convert to trt engine

The difference is that some modifications need to be made to the exported onnx file here. A `BatchedNMSDynamic_TRT` node is added to the end of the original graph, and the node attributes are adjusted according to the [TensorRT batchedNMSPlugin input format](https://github.com/NVIDIA/TensorRT/tree/main/plugin/batchedNMSPlugin#structure).

### 1.1 Modify before and after

- Original forward function output format in infer mode

  - squeezed boxes and classes:

    ```
    [batch_size, number_boxes, box_xywh + c + number_classes] = [batch_size, 25200, 85]
    ```

    ![](../assets/before.png)

- Modified output format

  - boxes

    ``` 
    [batch_size, number_boxes, 1, x1y1x2y2]
    ```
  
  - cls_conf

    ```
    [batch_size, number_boxes, number_classes] 
    ```

    ![](../assets/after.png)

    According to the [batchedNMSPlugin.cpp](https://github.com/NVIDIA/TensorRT/blob/main/plugin/batchedNMSPlugin/batchedNMSPlugin.cpp#L193) source code comments, the input shape of boxes should be `[batch_size, num_boxes, num_classes, 4]` or `[batch_size, num_boxes, 1, 4]`.

    The related explanation can be found in the [efficientNMSPlugin](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin#boxes-input) documentation:

    > The boxes input can have 3 dimensions in case a single box prediction is produced for all classes (such as in EfficientDet or SSD), or 4 dimensions when separate box predictions are generated for each class (such as in FasterRCNN), in which case number_classes >= 1 and must match the number of classes in the scores input. The final dimension represents the four coordinates that define the bounding box prediction.
    
    Since YOLOv5 is used, bounding boxes will not be generated for each category, so the input shape of boxes should be `[batch_size, num_boxes, 1, 4]`.

### 1.2 Modify detect layer

Change the output of the yolov5 Detect layer forward function to the [TensorRT batchedNMSPlugin input format](https://github.com/NVIDIA/TensorRT/tree/main/plugin/batchedNMSPlugin#structure).

```python
def forward(self, x):
    z = []  # inference output
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
        bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:  # inference
            if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            y = x[i].sigmoid()
            if self.inplace:
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                # custom output >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                conf = y[..., 4:]
                xmin = xy[..., 0:1] - wh[..., 0:1] / 2
                ymin = xy[..., 1:2] - wh[..., 1:2] / 2
                xmax = xy[..., 0:1] + wh[..., 0:1] / 2
                ymax = xy[..., 1:2] + wh[..., 1:2] / 2
                obj_conf = conf[..., 0:1] 
                cls_conf = conf[..., 1:]
                cls_conf *= obj_conf
                # y = torch.cat((xy, wh, y[..., 4:]), -1)
                y = torch.cat((xmin, ymin, xmax, ymax, cls_conf), 4)
            # z.append(y.view(bs, -1, self.no))
            z.append(y.view(bs, -1, self.no - 1))
    
    z = torch.cat(z, 1) 
    bbox = z[..., 0:4].view(bs, -1, 1, 4)
    cls_conf = z[..., 4:]
    
    return bbox, cls_conf
    # return x if self.training else (torch.cat(z, 1), x)
    # custom output >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
```

### 1.3 Modify export_onnx

When exporting onnx, modify the output to meet the [TensorRT batchedNMSPlugin input format](https://github.com/NVIDIA/TensorRT/tree/main/plugin/batchedNMSPlugin#structure).

Key points are introduced here, see [export_onnx function in export.py](../export.py) for detailed code.

- Avoid exporting as static shape during onnx simplify

  ```python
  model_onnx, check = onnxsim.simplify(
      model_onnx, 
      dynamic_input_shape=dynamic
      # must comment out  
      #input_shapes={'images': list(im.shape)} if dynamic else None
      )
  ```

- Use onnx-graphsurgeon to create a `BatchedNMSDynamic_TRT` node and append it to the end of the original graph

  ```python
  # add batch NMS:
  yolo_graph = onnx_gs.import_onnx(model_onnx)
  box_data = yolo_graph.outputs[0]
  cls_data = yolo_graph.outputs[1]
  nms_out_0 = onnx_gs.Variable(
      "BatchedNMS",
      dtype=np.int32
  )
  nms_out_1 = onnx_gs.Variable(
      "BatchedNMS_1", 
      dtype=np.float32
  )
  nms_out_2 = onnx_gs.Variable(
      "BatchedNMS_2",  
      dtype=np.float32
  )
  nms_out_3 = onnx_gs.Variable(
      "BatchedNMS_3",
      dtype=np.float32
  )
  nms_attrs = dict()
  # ........

  nms_plugin = onnx_gs.Node(
      op="BatchedNMSDynamic_TRT",
      name="BatchedNMS_N", 
      inputs=[box_data, cls_data],
      outputs=[nms_out_0, nms_out_1, nms_out_2, nms_out_3],
      attrs=nms_attrs
  )
  yolo_graph.nodes.append(nms_plugin)
  yolo_graph.outputs = nms_plugin.outputs
  yolo_graph.cleanup().toposort()
  model_onnx = onnx_gs.export_onnx(yolo_graph)
  ```

- Export onnx and tensorrt engine sequentially

  ```shell
  # export onxx 
  python export.py --weights yolov5s.pt --include onnx --simplify --dynamic

  # export trt engine
  /usr/src/tensorrt/bin/trtexec  \
  --onnx=yolov5s.onnx \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \ 
  --maxShapes=images:1x3x640x640 \
  --workspace=4096 \
  --saveEngine=yolov5s_opt1_max1_fp16.engine \
  --shapes=images:1x3x640x640 \
  --verbose \
  --fp16 \
  > result-FP16-BatchedNMS.txt 
  ```

---

## 2. Performance Testing

### 2.1 COCO17 validation dataset

Compare infer + nms elapsed time

- Original yolov5

  ```shell
  python detect.py --weight original-yolov5s-fp16.engine --half --img 640 --source </path/to/coco/images/val2017/> --device 0
  ```

  Speed: 0.8ms pre-process, 4.4ms inference, 2.2ms NMS per image at shape (1, 3, 640, 640)

- batchedNMSPlugin

  ```shell
  python trt_infer.py --model yolov5s_opt1_max1_fp16.engine --input_images_folder </path/to/coco/images/val2017/> --output_images_folder <output_tempfolder> --input_size 640
  ```

  Inference + NMS: 5.4 ms per image at shape (1, 3, 640, 640)

### 2.2 trtexec

trtexec is local test result. With batch size = 1, the overall difference is not significant. After integrating NMS into trt engine, the output tensor is much smaller, which can reduce device to host data transfer time. The cost is that GPU compute time is increased.

| Metrics | BatchedNMSDynamic_TRT engine<br>infer+nms | ultralytics engine<br>only infer |  
| :-: | :-: | :-: |
| Latency | 3.97021 ms | 4.08145 ms |
| End-to-End Host Latency | 6.70715 ms | 4.73285 ms |
| Enqueue Time | 1.27597 ms | 0.95929 ms |  
| H2D Latency | 0.563791 ms | 0.316406 ms |
| GPU Compute Time | 3.45068 ms | 2.41992 ms |
| D2H Latency | 0.0100889 ms | 1.34198 ms |

---
## REFERENCES

- [Ultralytics Yolov5](https://github.com/ultralytics/yolov5.git)
- [Yolov5 GPU Optimization](https://github.com/NVIDIA-AI-IOT/yolov5_gpu_optimization.git)  
- [TensorRT BatchedNMSPlugin](https://github.com/NVIDIA/TensorRT/tree/main/plugin/batchedNMSPlugin)