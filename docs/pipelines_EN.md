# Deploy yolov5 Triton Pipelines

## 1. Why use Triton pipelines

It is well known that model services include not only GPU based Inference, but also preprocess and postprocess. Triton Pipelines are workflows that can combine different model services into a complete application. The same model service can also be used by different workflows.

Therefore, preprocess or postprocess can be deployed separately, and then connected with the infer module through Pipeline. The benefits of doing this are:

- Each submodule can apply for different types and sizes of resources separately, and be configured with different parameters, in order to maximize model serving efficiency while making full use of computing resources.

- Overhead of transferring intermediate tensors can be avoided, reducing the amount of data transferred over the network, and minimizing the number of requests that need to be sent to Triton. 

---

## 2. Triton Pipeline implementation methods

Nvidia Triton provides two ways to deploy Piplelines: Business Logic Scripting(BLS) and Ensemble. Below is a brief introduction of the two methods.

- [Ensemble](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models)  
A workflow formed by combining various models in the model repository. It is a pipeline scheduling strategy rather than a specific model. Ensemble is more efficient, but cannot incorporate conditional logic judgments. Data can only flow according to the fixed pipeline, suitable for scenarios with fixed pipeline structure.

  ![](../assets/ensemble.png)

- [BLS](https://github.com/triton-inference-server/python_backend#business-logic-scripting)
A special python backend that calls other model instances through python code. BLS is more flexible, can incorporate some logic and loops to dynamically combine different models and thus control the data flow direction. 
  ![](../assets/bls.png)

---

## 3. How to deploy Triton Pipelines  

One motivation for deploying process modules through Pipelines is to reduce the amount of data transmitted over the network. In object detection model services, both the raw_image on the input side and the candidate bboxes before nms have relatively large data volume. Therefore, a reasonable approach is to deploy the nms postprocess module separately through python backend, and connect the infer and nms modules through pipelines. The client only needs to do necessary resize and other preprocess operations on the raw_data.


### 3.1 Workflow

Pipeline configuration and python backend refer to [ensemble](../triton/model_repository/simple_yolov5_ensemble/) and [bls](../triton/model_repository/simple_yolov5_bls/) in Model Repository.

Workflow of the two deployment methods is as follows:

![](../assets/bls_ensemble.png)


### 3.2 BLS

- Data flow
  1. Send resized image to BLS model service through http/gRPC
  2. BLS service calls yolov5 tensorrt model service through C API
  3. Triton Server returns candidate bboxes to BLS service
  4. BLS service performs nms operation on candidate bboxes and returns final bboxes to client through http/gRPC
 

### 3.3 Ensemble

- Data flow
  1. Send resized image to ensemble model service through http/gRPC
  2. Ensemble model service copies yolov5 tensorrt output candidate bboxes to nms model service through memory 
  3. Ensemble model service returns bboxes after nms to client through http/gRPC

### 3.3 Notice

The output bboxes number is not fixed. There are usually three ways to handle this:

1. Pad bboxes, for example, specify output as `[batch_size, padding_count, xywh or xyxy]`, where pandding_count is determined according to actual scenario
2. Put model output results into a json, returned as `json string ([N, 1])` 
3. Use [decoupled response](https://github.com/triton-inference-server/python_backend#decoupled-mode) 

Padding is used here to solve this problem
```python
from torch.nn import functional as F
i = torchvision.ops.nms(boxes, scores, nms_threshold)
# padding boxes to 300
if i.shape[0] > max_det:  # limit detections
    i = i[:max_det]
bbox_pad_nums = max_det - i.shape[0]  
output_bboxes[xi] = F.pad(x[i], (0,0,0, bbox_pad_nums), value=0) 
```

---
## REFERENCES

- [Ultralytics Yolov5](https://github.com/ultralytics/yolov5.git)
- [Ensemble models](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models)  
- [Business Logic Scripting](https://github.com/triton-inference-server/python_backend#business-logic-scripting)
- [Triton Tutorials](https://space.bilibili.com/1320140761/channel/collectiondetail?sid=493256)