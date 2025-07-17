from PyQt5.QtCore import  QThread, pyqtSignal
from PyQt5.QtGui import QImage
from utils import cv_image_to_qlabel,COCO_CLASSES, COCO_COLOR_LIST
import enum
import time
import cv2 as cv
import onnxruntime as ort
import numpy as np
providers = ort.get_available_providers()


def scale_boxes(boxes, input_shape, target_shape):
    """
    Scale bounding boxes from input_shape to target_shape.
    boxes: list of [x1, y1, x2, y2]
    input_shape: (input_h, input_w)
    target_shape: (target_h, target_w)
    """
    input_h, input_w = input_shape
    target_h, target_w = target_shape
    scale_x = target_w / input_w
    scale_y = target_h / input_h
    scaled_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)
        scaled_boxes.append([x1, y1, x2, y2])
    return scaled_boxes


def get_class_name(class_id):
    return COCO_CLASSES.get(class_id, "Unknown")

def get_class_color(class_id):
    """
    Get the color for a given class ID.
    This function can be modified to return different colors based on the class ID.
    """    
    return COCO_COLOR_LIST.get(class_id, (255, 255, 255))  # Default to white if class_id not found

def preprocess_image_yolo8(image, input_size):
    """
    Preprocess the image for YOLO model input.
    """
    # Resize the image to the input size

    if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
        dsize = tuple(input_size)
    else:
        dsize = (input_size, input_size)
    

    resized_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    resized_image = cv.resize(resized_image, dsize)
    print("Resized image shape: ", resized_image.shape)  # Debugging line
    
    
    # Normalize the image
    normalized_image = resized_image.astype(np.float32) / 255.0 #scale the image to [0,1] range

    normalized_image = np.transpose(normalized_image, (2, 0, 1))  # Change to CHW format
    normalized_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension

    return normalized_image

def apply_nms_yolo8(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression (NMS) to filter out overlapping boxes.
    """
    indices = cv.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=iou_threshold)
    
    return indices.flatten() if len(indices) > 0 else []

def postprocess_detections_yolo8(image,detections, class_confidence_threshold = 0.5, iou_threshold=0.4):
    """
    Postprocess the raw detections from the YOLO model.
    """
   
    boxes = detections[:4, :].T # Assuming the first 4 rows are x_center, y_center, width, height

    print("boxes: ", boxes.shape)  # Debugging line
    scores = detections[4:, :].T  # shape: (8400, 80)  # Assuming the 5th row is the confidence score

    detection_count = boxes.shape[0]  # Number of detections

    normalized_boxes = []
    class_ids = []
    confidence_scores = []
    results = []

    for i in range(detection_count):
        class_id = np.argmax(scores[i])

        confidence = scores[i][class_id]  # Get the confidence score for the class with the highest score
        
        if confidence >= class_confidence_threshold:
        
            x_center,y_center,w,h = boxes[i]
            print("boxe before normalization: ",boxes[i])
            x1 = int(x_center - (0.5*w))
            y1 = int(y_center - (0.5*h))
            x2 = int(x_center + (0.5*w))
            y2 = int(y_center + (0.5*h))
            print("boxe after normalization: ",[x1, y1, x2, y2])
    
            normalized_boxes.append([x1, y1, x2, y2])  # Append the bounding box coordinates
            class_ids.append(class_id)
            confidence_scores.append(confidence)

    # normalized_boxes = scale_boxes(normalized_boxes, (640, 640), (1080, 1920))  # Scale boxes to target shape
    #apply NMS to filter out overlapping boxes
    indices = apply_nms_yolo8(normalized_boxes, confidence_scores, iou_threshold)

    og_size = (image.shape[1], image.shape[0])  # Original image size (height, width)
    image = cv.resize(image, (640, 640))  # Resize the image to
    for i in indices:
        print(f"Detection initial i {i}: Box={normalized_boxes[i]}, Class={class_ids[i]}, Confidence={confidence_scores[i]}")
        # i = i[0] if isinstance(i, np.ndarray) else i  # Ensure i is an integer index
        # print(f"Detection {i}: Box={normalized_boxes[i]}, Class={class_ids[i]}, Confidence={confidence_scores[i]}")

        x1, y1, x2, y2 = map(int, normalized_boxes[i])
        print(f"Detection final i {i}: Box=({x1},{y1},{x2},{y2}), Class={class_ids[i]}, Confidence={confidence_scores[i]}")

        cv.rectangle(image, (x1, y1), (x2, y2), get_class_color(class_ids[i]), 2)
        cv.putText(image, f'ID: {get_class_name(class_ids[i])}, Score: {confidence_scores[i]:.4f}', 
                   (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, get_class_color(class_ids[i]), 1)
    image = cv.resize(image, og_size)  # Resize the image back to original size
    # return the filtered detections
    return image

def postprocess_detections_yolo11(detections, class_confidence_threshold=0.5, iou_threshold=0.4, num_classes=80, mask_dim=32):
    preds, protos = detections
    preds = np.squeeze(preds, axis=0)   # (N, 4+1+num_classes+mask_dim)
    protos = np.squeeze(protos, axis=0) # (mask_dim, mask_h, mask_w)

    boxes = preds[:, :4]  # x_center, y_center, w, h
    objectness = preds[:, 4]
    class_scores = preds[:, 5:5+num_classes]
    mask_vectors_all = preds[:, 5+num_classes:5+num_classes+mask_dim]

    normalized_boxes = []
    class_ids = []
    confidence_scores = []
    mask_vectors = []
    results = []
    protos_keepers = []

    detection_count = preds.shape[0]
    for i in range(detection_count):
        # Get the class with the highest score
        class_id = np.argmax(class_scores[i])
        print(f"Class ID: {class_id}, Objectness: {objectness[i]}, Class Scores: {class_scores[i]}")  # Debugging line
        confidence = class_scores[i][class_id]

        if confidence < class_confidence_threshold:
            continue

        x, y, w, h = boxes[i]
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        # print(f"Detection {i}: Box=({x1},{y1},{x2},{y2}), Class={class_id}, Confidence={confidence}")
        normalized_boxes.append([x1, y1, x2, y2])  # Append the bounding box coordinates
        class_ids.append(class_id)
        confidence_scores.append(confidence)
        mask_vectors.append(mask_vectors_all[i])
        protos_keepers.append(protos)
        # results.append([normalized_boxes[-1], confidence_scores[-1], class_ids[-1], mask_vectors[-1]])

    normalized_boxes = scale_boxes(normalized_boxes, (640, 640), (1080, 1920))  # Scale boxes to target shape
    # Apply NMS to filter out overlapping boxes
    indices = apply_nms_yolo8(normalized_boxes, confidence_scores, iou_threshold)

    # Return the filtered detections
    for i in indices:
        results.append([normalized_boxes[i], confidence_scores[i], class_ids[i], mask_vectors[i], protos_keepers[i]])


    print(f"Postprocessed {len(results)} detections.")
    # print("Results:", results)

    return results # returns a list of detections in the format [[box, confidence, class_id, mask_vector, protos], ...]


def inpaint_yolo_results_yolo8(results):

    image = results[-1]  

    image = cv.resize(image, (640, 640))  # Resize the image to the target size if needed

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

    # image = cv.resize(image, (1080, 1920))  # Resize the image to the target size if needed
    
    return image  # Return the processed image



def inpaint_yolo_results_yolo11(results):
    image = results[-1]  # The last item is the image

    detection_count = len(results) - 1  # Exclude the last item which is the image

    for i in range(detection_count):
        detection = results[i]
        x, y, w, h = detection[0]  # Assuming the first four values are x, y, width, height
        class_id = int(detection[2])  # Assuming the fifth value is the class
        score = detection[1]  # Assuming the sixth value is the confidence score

        cv.rectangle(image, (x, y), (x + w, y + h), get_class_color(class_id), 2)
        cv.putText(image, f'ID: {get_class_name(class_id)}, Score: {score:.4f}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, get_class_color(class_id), 1)

    return image  # Return the processed image

def string_to_model_type(model_type_str):
    """
    Convert a string to a ModelType enum.
    """
    try:
        return ModelType[model_type_str.upper()]
    except KeyError:
        raise ValueError(f"Invalid model type: {model_type_str}. Available types: {[e.name for e in ModelType]}")

class ModelType(enum.Enum):

    YOLO_8_D = "model_zoo/best.onnx"
    YOLOY_11_S = "model_zoo/yolo11n-seg.onnx"

class VideoONNXWorker(QThread):
    image = pyqtSignal(QImage)

    def __init__(self,video_source=-1,model=ModelType.YOLO_8_D,fps=30,limit_fps=True,model_confidence_threshold=0.5, iou_threshold=0.4,
                 provider = 'CUDAExecutionProvider' if 'CUDAExecutionProvider' in providers else 'CPUExecutionProvider' ):
        super().__init__()

        self.device = provider
        self.video_source = video_source
        self.model_type = model
        self.model = ort.InferenceSession(self.model_type.value, providers=[self.device])
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
        self.valid_video_stream = False
        

        self.capture = cv.VideoCapture(self.video_source)
        
        if self.capture.isOpened():
            self.valid_video_stream = True


    def load_model(self, model_type):
        if self.running:
            self.running = False

        self.model_type = string_to_model_type(model_type)
        self.model = ort.InferenceSession(self.model_type.value, providers=[self.device])
        self.input_name = self.model.get_inputs()[0].name  # Get the input name of the model

        self.running = True

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

                    output_image = frame.copy()  # Keep a copy of the original image for output

                    # Preprocess the frame for the model
                    preprocessed_frame = preprocess_image_yolo8(frame, 640) 
                   
                    # Run the model inference
                    try:

                        if self.model_type == ModelType.YOLO_8_D:
                            output = self.model.run(None, {self.input_name: preprocessed_frame})[0]
                            output = np.squeeze(output)  # Remove batch dimension
                            output_image = postprocess_detections_yolo8(output_image,output, class_confidence_threshold=self.confidence_threshold,
                                                            iou_threshold=self.iou_threshold)
                        elif self.model_type == ModelType.YOLOY_11_S:
                            output = self.model.run(None, {self.input_name: preprocessed_frame})

                                                        
                            # write the output to a file for debugging
                            with open("output/yolo11_output.txt", "w") as f:
                                f.write(str(output))
                            
                            
                            detections = postprocess_detections_yolo11(output, class_confidence_threshold=self.confidence_threshold,
                                                            iou_threshold=self.iou_threshold)

                        # Append the original image to the detections for display
                        # detections.append(output_image)  # Append the original image to the detections

                        # draw the detections on the output image
                        # output_image = inpaint_yolo_results_yolo8(detections)

                        # output_image = cv.resize(output_image, (1920, 1080))  # Resize the output image to fit the label

                        qt_image = cv_image_to_qlabel(output_image)

                        self.image.emit(qt_image)
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

    def set_confidence_threshold(self, confidence_threshold):
        """
        Update the confidence threshold for the model.
        """
        self.confidence_threshold = confidence_threshold

    def set_iou_threshold(self, iou_threshold):
        """
        Update the IOU threshold for the model.
        """
        self.iou_threshold = iou_threshold

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