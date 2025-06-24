from custom_workers.base_worker import BaseImageWorker
import onnx
import onnxruntime as ort
print("ONNX Runtime version:", ort.__version__)
print("ort available providers:", ort.get_available_providers())
from utils import qimage_to_cv_image,cv_image_to_qimage
import numpy as np
import cv2 as cv    
import time

class UpscaleWorker(BaseImageWorker):
    input_size = (240,416)
    video_writer = cv.VideoWriter("upscale_worker_output.mp4", cv.VideoWriter_fourcc(*'mp4v'), 30, (1664,960),isColor=False)  # Adjust the output size as needed
    
    frame_count = 0
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
            self.frame_count += 1

            if self.frame_count > 300 == 0:  # Save every 30 frames
                self.video_writer.release()
                self.video_writer = cv.VideoWriter("upscale_worker_output" + str(self.frame_count) + ".mp4", cv.VideoWriter_fourcc(*'mp4v'), 30, (960, 1664)) 
                print(f"Saving frame {self.frame_count} to video file.")
            
        

    def _resize_input(self, image):
        image = cv.resize(image, (self.input_size[1], self.input_size[0]), interpolation=cv.INTER_CUBIC)
        return image

    def _init_onnx_runtime(self, model_type):
        self.model_path = None
        self.ort_session = None
        provider = ort.get_available_providers()
        print("Available ONNX Runtime providers: ", provider)
        self.device = "CUDAExecutionProvider" if "CUDAExecutionProvider" in provider else "CPUExecutionProvider"

        if model_type == "esrgan":
            self.model_path = "model_zoo/model.onnx"

    def start_session(self):
        print("starting onnx runtime session using : ",self.device)
        if self.ort_session is None:
            self.ort_session = ort.InferenceSession(self.model_path, providers=[self.device])


        return self.ort_session
    
    def feed_model(self, data):
        image = qimage_to_cv_image(data)
        source_shape = image.shape
        if source_shape[0] != self.input_size[1] or source_shape[1] != self.input_size[0]:
                image = self._resize_input(image)

        image_np = image.astype(np.float32)
        image_np = image_np.transpose(2, 0, 1)
        
        image_tensor = image_np[None,...]

        model_input = {self.ort_session.get_inputs()[0].name: image_tensor}

        start = time.perf_counter()

        try:
            model_output = self.ort_session.run(None,model_input)
            end = time.perf_counter()
            print(f"Model inference time: {end - start:.4f} seconds")
            output_image = model_output[0]

            print("Output shape: ", output_image.shape)
            output_image = output_image.squeeze(0).transpose(1, 2, 0)

            output_image = cv.cvtColor(output_image,cv.COLOR_RGB2GRAY)
            cv.imwrite("output/output_image_" +str(self.frame_count)+".png", output_image)
            self.video_writer.write(output_image.astype(np.uint8))

            print("Output shape after squeeze and transpose: ", output_image.shape)
        except Exception as e:
            print(f"Error during model inference: {e}")
            output_image = np.zeros((self.input_size[0], self.input_size[1]), dtype=np.uint8)
            self.video_writer.write(output_image)
            self.video_writer.release()    

        return cv_image_to_qimage(output_image.astype(np.uint8))
        
    
    def stop_thread(self):
        self.data = None
        self.video_writer.release()
        
        self.wait()
