import cv2 as cv
import onnxruntime as ort
import numpy as np
from PyQt5.QtCore import  QThread, pyqtSignal,pyqtSlot
from PyQt5.QtGui import QImage
from utils import qimage_to_cv_image, cv_image_to_qimage, cv_image_to_qlabel,COCO_CLASSES, COCO_COLOR_LIST
from custom_workers.camera_worker import CameraFeedMode

import time

providers = ort.get_available_providers()

def get_class_name(class_id):
    return COCO_CLASSES.get(class_id, "Unknown")


def get_class_color(class_id):
    """
    Get the color for a given class ID.
    This function can be modified to return different colors based on the class ID.
    """    
    return COCO_COLOR_LIST.get(class_id, (255, 255, 255))  # Default to white if class_id not found

def preprocess_image(image, input_size):
    """
    Preprocess the image for YOLO model input.
    """
    # Resize the image to the input size

    if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
        dsize = tuple(input_size)
    else:
        dsize = (input_size, input_size)
    resized_image = cv.resize(image, dsize)
    resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
    
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


def postprocess_detections(detections, class_confidence_threshold = 0.5, iou_threshold=0.4):
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


def inpaint_yolo_results(results):

    image = results[-1]  

    detection_count = len(results) - 1  # Exclude the last item which is the image
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert BGR to RGB for display

    for i in range(detection_count):# 
        detection = results[i]
        x, y, w, h = detection[0]  # Assuming the first four values are x, y, width, height
        class_id = int(detection[2])  # Assuming the fifth value is the class
        score = detection[1]  # Assuming the sixth value is the confidence score

        cv.rectangle(image, (x, y), (x + w, y + h), get_class_color(class_id), 2)
        # Optionally add text for class_id and score
        cv.putText(image, f'ID: {get_class_name(class_id)}, Score: {score:.4f}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, get_class_color(class_id), 1)


    # Convert the processed image back to QPixmap
    # processed_image = cv_image_to_qimage(image)

   
    
    return image  # Return the processed image


class VideoONNXWorker(QThread):
    image = pyqtSignal(QImage)

    def __init__(self,video_source=-1,model_source='model_zoo/yolov8n_det.onnx',fps=30,limit_fps=True,model_confidence_threshold=0.5, iou_threshold=0.4,
                 provider = 'CUDAExecutionProvider' if 'CUDAExecutionProvider' in providers else 'CPUExecutionProvider' ):
        super().__init__()

        print("Onnx provider:", provider)
        self.video_source = video_source
        self.model_source = model_source
        self.model = ort.InferenceSession(self.model_source, providers=[provider])
        self.running = False
        self.paused = False
        self.capture = None
        self.limited_fps = limit_fps  # Flag to indicate if FPS limiting is enabled
        self.fps = fps  # Frames per second limit
        self.wait_for_next = False  # Flag to indicate if we are waiting for the next frame in single frame mode
        self.single_frame_mode = False
        self.prev = None  # To store the previous frame for paused state
        self.confidence_threshold = model_confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_name = self.model.get_inputs()[0].name  # Get the input name of the model    
        # self.input_size = (self.model.get_inputs()[0].shape[2], self.model.get_inputs()[0].shape[3])  # Get the input size of the model
        # print("input size:", self.input_size)


        self.capture = cv.VideoCapture(self.video_source)
        # get the input size from the capture
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video source: {self.video_source}")



    def run(self):
        self.running = True
        prev_time = time.time()

        while self.running:
            
            if not self.paused:
                start = time.perf_counter()
                # FPS limiting logic
                if self.limited_fps and self.fps > 0:
                    current_time = time.time()
                    elapsed = current_time - prev_time
                    wait_time = max(0, (1.0 / self.fps) - elapsed)
                    if wait_time > 0:
                        self.msleep(int(wait_time * 1000))
                    prev_time = time.time()

                ret, frame = self.capture.read()

                if ret:
                    self.prev = cv_image_to_qlabel(frame)

                    # print("frame type:", type(frame))

                    output_image = frame.copy()  # Keep a copy of the original image for output

                    # Preprocess the frame for the model
                    preprocessed_frame = preprocess_image(frame, 640)  # Assuming the model input size is (416, 240)
                    # Run the model inference
                    try:
                        output = self.model.run(None, {self.input_name: preprocessed_frame})[0]
                        output = np.squeeze(output)  # Remove batch dimension

                        detections = postprocess_detections(output, class_confidence_threshold=self.confidence_threshold,
                                                            iou_threshold=self.iou_threshold)
                        
                        # Append the original image to the detections for display
                        detections.append(output_image)  # Append the original image to the detections

                        # draw the detections on the output image
                        output_image = inpaint_yolo_results(detections)
                    
                        qt_image = cv_image_to_qlabel(output_image)

                        self.image.emit(qt_image)
                        print(f"Frame processed in {time.perf_counter() - start:.4f} seconds that is {1 / (time.perf_counter() - start):.2f} FPS")
                    except Exception as e:
                        print(f"Error during model inference: {e}")
                        output_image = cv_image_to_qlabel(output_image)
                        self.image.emit(output_image)  # Emit the frame without detections in case of error
                        continue
                else:
                    break
            else:
                # If paused, emit the previous frame
                #convert the previous frame to QLabel format
                output_image = cv_image_to_qlabel(self.prev) if self.prev is not None else QImage()
                self.image.emit(output_image)
                self.msleep(100)

        self.capture.release()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def toggle_single_frame_mode(self, enable):
        self.single_frame_mode = enable

    def set_video_sorce(self, video_path):
        if self.running:
            self.running = False
            self.wait()  # Wait for the thread to finish before changing the video path
        
        self.capture.release()
        self.capture = cv.VideoCapture(video_path)  # Start a new capture with the new
        self.fps = self.capture.get(cv.CAP_PROP_FPS)
        
        self.running = True
        self.start()

    def stop(self):
        self.running = False
        self.capture.release()
        self.wait()