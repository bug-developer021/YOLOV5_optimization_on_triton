
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
        # output_dtype = self.output_dtype
        max_det = self.max_det
        responses = []
        for request in requests:

            before_nms = pb_utils.get_input_tensor_by_name(
                request, 'candidate_boxes')

            # before_nms_torch_tensor = self.pb_tensor_transform(before_nms)
            print (f'nms pb_tensor is from cpu {before_nms.is_cpu()}', flush=True)
            before_nms_torch_tensor = from_dlpack(before_nms.to_dlpack())

            
            bboxes = utils.postprocess(before_nms_torch_tensor, max_det=max_det)

            # print(f'bls bboxes shape is {bboxes.shape}', flush=True)

            # encoding pytorch tensor boxes  to pb_tensor
            # out_tensor = pb_utils.Tensor('BBOXES', bboxes.astype(output_dtype))
            out_tensor = pb_utils.Tensor.from_dlpack('BBOXES', to_dlpack(bboxes))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses



    def finalize(self):
        print('Cleaning up...')

