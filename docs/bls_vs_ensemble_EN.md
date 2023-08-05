# Comparison of Triton Pipeline implementation methods 

In [Deploying yolov5 Triton Pipelines](pipelines_EN.md#2-triton-pipeline-implementation-methods), BLS and Ensemble, two ways of implementing Triton Pipelines, are briefly introduced. In [Benchmark](../README.md#benchmark), the three deployment methods of `BLS Pipelines`, `Ensemble Pipelines`, and [All in TensorRT Engine](./batchedNMS_EN.md) are performance tested under gradually increasing concurrency. This article will compare and introduce BLS and Ensemble, and interpret the performance test results.

## 1 Python Backend

### 1.1 Implementation and structure
BLS is a special python backend that completes Pipelines by calling other model services in python backend. The structure of python backend is as follows:

![](../assets/python_backend.png)

- Inter-process communication IPC

  Due to GIL limitations, python backend supports multi-instance deployment by starting a separate process (`python stub process(C++)`) for each model instance. Since it is multi-process, `shared memory` is used to complete the communication between the python model instance and the Triton main process. Specifically, a shm block is allocated in the `shared memory` for each python stub process, and the shm block connects the `python backend agent(C++)` for communication.

- Data flow

  `shm block` schedules and forwards Input and Output through `Request MessageQ` and `Response MessageQ`. Both queues are implemented using producer-consumer model logic.
  1. The request sent to Triton server is put into `Request MessageQ` by `python backend agent(C++)`
  2. The python stub process takes the Input from the `Request MessageQ`, passes it to the python model instance for inference, and then puts the Output into the `Response MessageQ`
  3. `python backend agent(C++)` takes the Output from the `Response MessageQ` and packages it into a response returned to Triton server main process

  For example:
  ```python
  responses = []
  for request in requests:
      input_tensor = pb_utils.get_input_tensor_by_name(
          request, 'input')
  
      # INFER_FUNC is python backend core logic
      output_tensor = INFER_FUNC(input_tensor)
  
      inference_response = pb_utils.InferenceResponse(
          output_tensors=[out_tensor])
      responses.append(inference_response)
  ```

### 1.2 Notice

- Need to manually manage whether Tensors are on CPU or GPU, `instance_group {kind: KIND_GPU}` in config does not work
- Input is not automatically batched, requests list needs to be manually converted to batch, same for all backends
- By default, python backend actively moves input tensor to CPU before inference, set `FORCE_CPU_ONLY_INPUT_TENSORS` to `no` to avoid host-device memory copies as much as possible 
- Python backend model instance exchanges data with Triton server through shared memory, so each instance requires a large shared memory, at least 64MB
- If performance becomes a bottleneck, especially with many loops, switch to C++ backend

---

## 2 BLS
A special python backend that calls other model services through python code. Use cases: dynamically combine deployed model services through some logic judgments.

### 2.1 BLS workflow

![](../assets/bls_arc.png)

The part above the dotted line is the general way to call the python backend. The part below the dotted line is to call other model services in the python backend. The overall workflow can be summarized as:

1. The python model instance processes the received Input tensor
2. The python model instance initiates a request through BLS call
3. The request goes through the python stub process into the shm block
4. The python backend agent takes the BLS input from the shm block and sends it to the specified model for inference through Triton C API
5. The Triton python backend agent sends the inferred output to the shm block
6. The BLS Output goes through the python stub process, taken from the shm block, packaged into a BLS response, and returned to the python model instance

### 2.2 Notice

- Location of Input tensor  
  By default, python backend actively moves input tensor to CPU before providing it for inference. Set `FORCE_CPU_ONLY_INPUT_TENSORS` to `no` to avoid this behavior. The location of the input tensor depends on how it is finally processed. After enabling this setting, the python backend needs to be able to handle both CPU and GPU tensors at the same time.

- Execution order of modules
  BLS does not support step parallelism, steps must be executed sequentially, the next step is executed only after the previous step is completed.

- Data transfer
  Use `DLPack` for tensor encoding/decoding between different frameworks and python backend. This step has zero copy and is very fast.

---

## 3 Ensemble

### 3.1 Overview of Ensemble
Using Ensemble to implement Pipelines can avoid the overhead of intermediate tensor transfer and minimize the number of requests that must be sent to Triton server. Compared to BLS, the advantage of Ensemble is that it can parallelize the execution of multiple models (steps), thereby improving overall performance. 

A typical Ensemble Pipeline is as follows:
```
name: "simple_yolov5_ensemble"
platform: "ensemble" 
max_batch_size: 8
input [
  {
    name: "ENSEMBLE_INPUT_0"
    data_type: TYPE_FP32
    dims: [3, 640, 640]
  }
]

output [
  {
    name: "ENSEMBLE_OUTPUT_0" 
    data_type: TYPE_FP32
    dims: [ 300, 6 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "simple_yolov5"
      model_version: 1
      input_map: {
        key: "images"
        value: "ENSEMBLE_INPUT_0"
      }
      output_map: {
        key: "output"
        value: "FILTER_BBOXES" 
      }
    },
    {
      model_name: "nms"
      model_version: 1
      input_map: {
        key: "candidate_boxes"
        value: "FILTER_BBOXES"
      }
      output_map: {
        key: "BBOXES"
        value: "ENSEMBLE_OUTPUT_0"  
      }
    }
  ]
}
```
The above Pipeline contains two independently deployed model services [simple_yolov5](../triton/model_repository/simple_yolov5/config.pbtxt) and [nms](../triton/model_repository/nms/config.pbtxt) connected by Ensemble. The output of simple_yolov5 is the input of nms, and the output of nms is the output of the entire Pipeline. Each input_map and output_map is a key-value pair, where key is the input/output name of each model service, and value is the input/output name of Ensemble.

### 3.2 Ensemble data transfer

- If all child models of Ensemble are deployed based on Triton built-in framework backends, data between child models can be transferred point-to-point via CUDA API without CPU memory copy.

- If child models of Ensemble use custom backends or python backends, tensor communication between child models is completed by system (CPU) memory copy, even if `FORCE_CPU_ONLY_INPUT_TENSORS` is set to `no` in python backend. As in the following step, the output of the previous step is from tensorrt backend on GPU, but the input printed in python backend is always on CPU, meaning a Device to Host memory copy happened here.

    ```python
    for request in requests:

        before_nms = pb_utils.get_input_tensor_by_name(
            request, 'candidate_boxes')

        # always true
        print (f'nms pb_tensor is from cpu {before_nms.is_cpu()}', flush=True) 
    ```
---

## 4 Performance analysis

Data source: [Benchmark](../README.md#benchmark) 

Throughput and latency are the two main performance metrics considered. Latency difference between the three is not big, but in terms of throughput, `batched_nms_dynamic > Ensemble > BLS`. The reasons are:

- inference and nms are all included in the trt engine for batched_nms_dynamic, communication between layers is via CUDA API, which is most efficient
- For Ensemble and BLS, inference and nms are two separate model instances. For BLS, the input tensor is on GPU in python backend, while for Ensemble the input tensor is forced to CPU, the overhead of memory copy outweighs the benefits of step parallelism. Therefore, when python backend is involved, BLS performs better than Ensemble