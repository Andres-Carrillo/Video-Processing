import cv2 as cv
import onnxruntime as ort
import numpy as np
from PyQt5.QtCore import  QThread, pyqtSignal,pyqtSlot
from PyQt5.QtGui import QImage
from custom_workers.base_worker import BaseImageWorker
from utils import qimage_to_cv_image, cv_image_to_qimage
import queue

def preprocess_image(image, input_size):
    """
    Preprocess the image for YOLO model input.
    """
    # Resize the image to the input size
    resized_image = cv.resize(image, (input_size, input_size))
    resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Normalize the image
    normalized_image = resized_image.astype(np.float32) / 255.0

    cv.imwrite("preprocessed_image.jpg", normalized_image*255)  # Save preprocessed image for debugging
    normalized_image = np.transpose(normalized_image, (2, 0, 1))  # Change to CHW format
    normalized_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension


    return normalized_image


def apply_nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression (NMS) to filter out overlapping boxes.
    """
    indices = cv.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=iou_threshold)
    
    return indices.flatten() if len(indices) > 0 else []


def postprocess_detections(detections, input_size, class_confidence_threshold = 0.5,obj_conf_threshold=0.2, iou_threshold=0.4):
    """
    Postprocess the raw detections from the YOLO model.
    """

    boxes = detections[:4, :].T # Assuming the first 4 rows are x_center, y_center, width, height
    scores = detections[4:, :].T  # shape: (8400, 80)  # Assuming the 5th row is the confidence score
    
    detection_count = scores.shape[0]  # Number of detections

    normalized_boxes = []
    class_ids = []
    confidence_scores = []
    results = []

    for i in range(detection_count):
        class_id = np.argmax(scores[i])
        confidence = scores[i][class_id]  # Get the confidence score for the class with the highest score
        if confidence < class_confidence_threshold:
            continue

        x,y,w,h = boxes[i]
        x1 = int(x -w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
  
        normalized_boxes.append([x1, y1, x2, y2])  # Append the bounding box coordinates
        class_ids.append(class_id)
        confidence_scores.append(confidence)


    #apply NMS to filter out overlapping boxes
    indices = apply_nms(normalized_boxes, confidence_scores, iou_threshold)

    # return the filtered detections
    return [[normalized_boxes[i], confidence_scores[i], class_ids[i]] for i in indices]

class YOLOWorker(QThread):
        results = pyqtSignal(list)
   
        def __init__(self,model_path:str='',input_size:int=640,
                     conf_threshold:float=0.5, iou_threshold:float=0.4):
            
            super().__init__()
            self.model_path = model_path
            self.input_size = input_size
            self.conf_threshold = conf_threshold
            self.iou_threshold = iou_threshold
            
            # Load the ONNX model
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.running = False
            self.processing_list = queue.Queue(maxsize=100)  # Initialize the processing list
            self.paused = False  # Flag to indicate if the worker is paused

        def start(self):
            """
            Start the worker thread.
            """
            self.running = True
            super().start()


        def run(self):
                """
                Run the worker thread to process images.
                This method is called when the thread starts.
                """
                while self.running:
                    if self.processing_list.empty() or self.paused:
                        self.msleep(100)
                        continue

                    # Get the next frame from the processing list
                    data = self.processing_list.get()

                    if data is None:  # Check for termination signal
                        break

                    # Convert QImage to numpy array
                    image = qimage_to_cv_image(data)
                    output_image =image.copy()  # Keep a copy of the original image for output

                    # Check if the image is valid
                    if image is None:
                        self.results.emit([])
                        continue

                    # Preprocess the image
                    preprocessed_image = preprocess_image(image, self.input_size)

                   
                    
                    try:
                        # Run inference

                        outputs = self.session.run(None, {self.input_name: preprocessed_image})
                        output = outputs[0]  # Assuming the model outputs a single tensor
                        output = np.squeeze(output)  # Remove batch dimension
                    except Exception as e:
                        print(f"Error during model inference: {e}")
                        self.results.emit([])
                        continue
                    # Postprocess the detections
                    detections = postprocess_detections(output, self.input_size, self.conf_threshold, self.iou_threshold)
                    
                    # add output image to end of detections
                    detections.append(output_image)

                    # cv.imwrite("output_image.jpg", output_image)  # Save output image for debugging
                    # Emit the results
                    self.results.emit(detections)

        def stop_thread(self):
            """
            Stop the worker thread.
            """
            self.running = False
            self.processing_list.put(None)
            self.quit()
            self.wait()
            del self.session

        def add_frame(self, frame):
            """
            Add a frame to the processing list.
            """
            if not self.processing_list.full():
                self.processing_list.put(frame)
            else:
                print("Processing list is full. Frame dropped.")


        def clear_processing_list(self):
            """
            Clear the processing list.
            """
            while not self.processing_list.empty():
                try:
                    self.processing_list.get_nowait()
                except queue.Empty:
                    break


        def pause(self):
            """
            Pause the worker thread.
            """
            self.paused = True

        def resume(self):
            """
            Resume the worker thread.
            """
            self.paused = False
