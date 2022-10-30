
import json

import  triton_python_backend_utils as pb_utils
import numpy as np
from torch.utils.dlpack import from_dlpack, to_dlpack
import  utils




class TritonPythonModel:

    def initialize(self, args):

        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        print('Initializing...')
        self.model_config = model_config = json.loads(args['model_config'])
        output_config = pb_utils.get_output_config_by_name(model_config, "BBOXES")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config['data_type'])
        self.max_det = output_config['dims'][0]

        # print(f'output_dims {self.output_dims} type is {type(self.output_dims)}', flush=True)

    def execute(self, requests):
        output_dtype = self.output_dtype
        max_det = self.max_det
        responses = []
        for request in requests:

            # Get Model Name
            # hard code temporarily
            model_name_string = 'simple_yolov5'

            # model_name = pb_utils.get_input_tensor_by_name(
            #     request, 'MODEL_NAME')

            # Model Name string
            # batch_size = 1
            # model_name_string = model_name.as_numpy()[0].item()
        
            input = pb_utils.get_input_tensor_by_name(
                request, 'images')

            
            # for fix TritonModelException:
            t1 = utils.time_sync()
            before_nms = self.request_real_engine(
                input, model_name_string)
            t2 = utils.time_sync()

           

            print(f'bls request_real_engine time: {(t2-t1)*1000} ms', flush=True)
            before_nms_torch_tensor = self.pb_tensor_transform(before_nms)
            
            t3 = utils.time_sync()
            print(f'bls pb_tensor_transform time: {(t3-t2)*1000} ms', flush=True)
            bboxes = utils.postprocess(before_nms_torch_tensor, max_det=max_det)
            t4 = utils.time_sync()
            # 3~10ms
            print(f'bls postprocess time: {(t4 - t3)*1000} ms', flush=True)
            # print(f'bls bboxes shape is {bboxes.shape}', flush=True)

            # encoding pytorch tensor boxes  to pb_tensor
            # out_tensor = pb_utils.Tensor('BBOXES', bboxes.astype(output_dtype))
            out_tensor = pb_utils.Tensor.from_dlpack('BBOXES', to_dlpack(bboxes))
            # false
            # print(f'out_tensor is on cpu: {out_tensor.is_cpu()}', flush=True)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor])
            responses.append(inference_response)
            # t3 = utils.time_sync()
            # print(f'output time: {(t3 - t2)*1000} ms', flush=True)
        return responses

    # BLS
    def request_real_engine(self, frames_tensor, model_name_string):
        # frames_tensor: tensor

        inference_request = pb_utils.InferenceRequest(
            model_name=model_name_string,
            requested_output_names=['output'],
            inputs=[frames_tensor]
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())

        # tensor
        before_nms = pb_utils.get_output_tensor_by_name(inference_response, 'output')

        print (f'bls pb_tensor is from cpu {before_nms.is_cpu()}', flush=True)
         
        return before_nms


    def finalize(self):
        print('Cleaning up...')


    def pb_tensor_transform(self, pb_tensor):
        if pb_tensor.is_cpu():
            # print(f'bls pb_tensor is from cpu', flush=True)
            return pb_tensor.as_numpy()
        else:
            pytorch_tensor = from_dlpack(pb_tensor.to_dlpack())
            # print(f'bls pb_tensor is from {pytorch_tensor.device}', flush=True)
            return pytorch_tensor
            # return pytorch_tensor.cpu().numpy()