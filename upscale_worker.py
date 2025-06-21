from custom_workers.base_worker import BaseImageWorker
import onnx
import onnxruntime as ort
from utils import qimage_to_cv_image,cv_image_to_qimage
import numpy as np
import cv2 as cv    

class UpscaleWorker(BaseImageWorker):
    def __init__(self, model_type,is_deep_learning=True):
        super().__init__()
        
        if is_deep_learning: 
            self._init_onnx_runtime(model_type)
        else:
            self.model_path = None
            self.ort_session = None
            self.cv_method = 'bicubic'  # Placeholder for non-deep learning methods

        
        self.worker_name = "UpscaleWorker"


    def process_data(self, data):

        # If no model is provided, use a simple bicubic interpolation
        if self.model_path is None:            
            image = qimage_to_cv_image(data)
            self.data = cv_image_to_qimage(cv.resize(image, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC))
        else:
            # if a model is provided, feed it and set the class variable self.data 
            # so that it can be emitted in the process method inhereted from BaseImageWorker  
            if self.ort_session is None:
                self.start_session()
            
            self.data = self.feed_model(data)
        
    def _init_onnx_runtime(self, model_type):
        self.model_path = None
        self.ort_session = None
        provider = ort.get_available_providers()
        self.device = "CUDAExecutionProvider" if "CUDAExecutionProvider" in provider else "CPUExecutionProvider"

        if model_type == "esrgan":
            self.model_path = "model_zoo/model.onnx"

    def start_session(self):
        if self.ort_session is None:
            self.ort_session = ort.InferenceSession(self.model_path, providers=[self.device])

        return self.ort_session
    
    def feed_model(self, data):
        image = qimage_to_cv_image(data)

        image_np = image.astype(np.float32)
        image_np = image_np.transpose(2, 0, 1)
        
        image_tensor = image_np[None,...]

        model_input = {self.ort_session.get_inputs()[0].name: image_tensor}
        model_output = self.ort_session.run(None,model_input)

        output_image = model_output[0]
        output_image = output_image.squeeze(0).transpose(1, 2, 0)

        return cv_image_to_qimage(output_image.astype(np.uint8))
        
    
        