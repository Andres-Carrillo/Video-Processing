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
    bgr_color = COCO_COLOR_LIST.get(class_id, (255, 255, 255))  # Default to white if class_id not found
    return (bgr_color[2], bgr_color[1], bgr_color[0])  # Convert RGB to BGR

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



def convert_top_left_box_to_xyxy(box):
    """
    Convert bounding boxes from top-left format (x1, y1, width, height) to xyxy format (x1, y1, x2, y2).
    boxes: array of shape (N, 4) where each row is [x1, y1, width, height]
    """
    x1, y1, width, height = box
    x2 = x1 + width
    y2 = y1 + height

    return np.array([x1, y1, x2, y2])

def apply_area_based_nms(boxes, scores, iou_threshold):
    """"
    Apply Non-Maximum Suppression (NMS) based on the area of the bounding boxes.
    Sorting boxes by area and applying NMS. Inorder to merge enclosed boxes.
    """

    # Compute the area of each box the boxes are in xy width height format
    areas = [box[2]  * box[3] for box in boxes]
    # Sort boxes by area (largest first)    
    sorted_indices = np.argsort(areas)[::-1]
    sorted_boxes = [boxes[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]


    keep = []
    for i,temp_sorted in enumerate(sorted_boxes):
        print("i: ", i)  # Debugging line

        if len(keep) == 0:
            temp_sorted = convert_top_left_box_to_xyxy(temp_sorted)
            keep.append(temp_sorted)
            continue
        
        for j, temp_keep in enumerate(keep):
            temp_area = (temp_sorted[2] * temp_sorted[3])
            keep_area = (temp_keep[2] - temp_keep[0]) * (temp_keep[3] - temp_keep[1] + 1)

            area_of_union = temp_area + keep_area
    
            converted_temp = convert_top_left_box_to_xyxy(temp_sorted)
            converted_keep = convert_top_left_box_to_xyxy(temp_keep)

            # find the intersection box
            x1 = max(converted_temp[0], converted_keep[0])
            y1 = max(converted_temp[1], converted_keep[1])
            x2 = min(converted_temp[2], converted_keep[2])
            y2 = min(converted_temp[3], converted_keep[3])

            intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


            iou = intersection_area / (area_of_union - intersection_area) if area_of_union - intersection_area > 0 else 0


            if iou > iou_threshold:
                # Merge the boxes by taking the min and max coordinates
                merged_box = [
                    min(temp_sorted[0], temp_keep[0]),
                    min(temp_sorted[1], temp_keep[1]),
                    max(temp_sorted[2], temp_keep[2]),
                    max(temp_sorted[3], temp_keep[3])
                ]

                keep[j] = merged_box
                break

        else:
            # If no overlap was found, add the current box to keep
            keep.append(sorted_boxes[i])

        return  keep


def center_to_xyxy(box):
    """
    Convert bounding boxes from center format (x_center, y_center, width, height) to xyxy format (x1, y1, x2, y2).
    boxes: array of shape (N, 4) where each row is [x_center, y_center, width, height]
    """
    x_center, y_center, width, height = box
    x1 = x_center - width * 0.5
    y1 = y_center - height * 0.5
    x2 = x_center + width * 0.5
    y2 = y_center + height * 0.5

    return  np.array([x1, y1, x2, y2])


def scale_box(box, input_shape, target_shape):
    """
    Scale a single bounding box from input_shape to target_shape.
    box: [x1, y1, x2, y2]
    input_shape: (input_h, input_w)
    target_shape: (target_h, target_w)
    """
    input_h, input_w = input_shape
    target_h, target_w = target_shape
    scale_x = target_w / input_w
    scale_y = target_h / input_h
    x1, y1, x2, y2 = box
    x1 = int(x1 * scale_x)
    x2 = int(x2 * scale_x)
    y1 = int(y1 * scale_y)
    y2 = int(y2 * scale_y)
    return [x1, y1, x2, y2]


def postprocess_detections_yolo8(detections, class_confidence_threshold = 0.5, iou_threshold=0.4):
    """
    Postprocess the raw detections from the YOLO model.
    """

    print("ouput from detection model: ", detections.shape)  # Debugging line
   
    boxes = detections[:4, :].T # Assuming the first 4 rows are x_center, y_center, width, height

    # print("boxes: ", boxes.shape)  # Debugging line
    scores = detections[4:, :].T  # shape: (8400, 80)  # Assuming the 5th row is the confidence score

    detection_count = boxes.shape[0]  # Number of detections

    normalized_boxes = []
    class_ids = []
    confidence_scores = []

    for i in range(detection_count):
        class_id = np.argmax(scores[i])

        confidence = scores[i][class_id]  # Get the confidence score for the class with the highest score
        
        if confidence >= class_confidence_threshold:
    
            normalized_boxes.append(center_to_xyxy(boxes[i]))  # Append the bounding box coordinates
            class_ids.append(class_id)
            confidence_scores.append(confidence)


    #apply NMS to filter out overlapping boxes
    indices = apply_nms_yolo8(normalized_boxes, confidence_scores, iou_threshold)


    # return the filtered detections
    return [[normalized_boxes[i], confidence_scores[i], class_ids[i]] for i in indices]



def inpant_yolo8_detections(image, results):
    for box, score, class_id in results:
        x1, y1, x2, y2 = map(int, box)
        x1, y1, x2, y2 = scale_box([x1, y1, x2, y2], (640, 640), image.shape[:2])  # Scale the box to the original image size

        cv.rectangle(image, (x1, y1), (x2, y2), get_class_color(class_id), 2)
        cv.putText(image, f'ID: {get_class_name(class_id)}, Score: {score:.4f}',
                   (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, get_class_color(class_id), 1)

    # return the filtered detections
    return image


def postprocess_detections_yolo11(image, preds, protos, class_confidence_threshold=0.5, iou_threshold=0.4, num_classes=80, mask_dim=32):
    preds = np.squeeze(preds, axis=0)   # (N, 4+num_classes+mask_dim)
    preds = preds.T # Transpose to (4+num_classes+mask_dim, N)
    protos = np.squeeze(protos, axis=0) # (mask_dim, mask_h, mask_w)

    # 1. Split preds into boxes, class scores, mask coefficients
    boxes = preds[:, :4]  # xyxy or xywh depending on model
    scores = preds[:, 4:4+num_classes]  # class scores
    mask_dim = protos.shape[0]
    mask_coeffs = preds[:, -mask_dim:] # mask coefficients 



    # 2. For each detection, get best class and confidence
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # 3. Filter by confidence threshold
    keep = confidences > class_confidence_threshold
    boxes = boxes[keep]
    class_ids = class_ids[keep]
    confidences = confidences[keep]
    mask_coeffs = mask_coeffs[keep]


    # # 5. (Optional) Apply NMS (implement or use OpenCV/NumPy NMS)
    # indices = apply_nms_yolo8(boxes, confidences, iou_threshold)
    # boxes, class_ids, confidences, mask_coeffs = boxes[indices], class_ids[indices], confidences[indices], mask_coeffs[indices]

    # 6. Generate masks for each detection
    masks = []
    for coeff in mask_coeffs:
        mask = np.tensordot(coeff, protos, axes=([0], [0]))  # shape: (mask_h, mask_w)
        mask = 1 / (1 + np.exp(-mask))  # sigmoid
        mask = cv.resize(mask, (image.shape[1], image.shape[0]))  # resize to image size
        mask = (mask > 0.5).astype(np.uint8)  # threshold
        masks.append(mask)


    filtered_boxes = []
    filtered_class_ids = []
    filtered_confidences = []
    filtered_masks = []
    class_masks = {}

    detections = {}


    # group detections
    detections = zip(boxes, class_ids, confidences, masks)

    #sort detection by class id
    class_sort = sorted(detections, key=lambda x: x[1])  # Sort by class_id


    #iterate through sorted detections,
    # and bitwise or the masks of the same class
    # then find bounding boxes for masks based on contours
    # then apply nms of class bounding boxes
    # and draw the boxes, labels, and masks on the image

    # 6. Create a dictionary to hold masks for each class

    # 7. Draw boxes, labels, and masks on the image
    for box, class_id, conf, mask in class_sort:

        if class_id not in class_masks.keys():
            class_masks[class_id] = np.zeros_like(image, dtype=np.uint8)

        class_masks[class_id] = cv.add(class_masks[class_id], mask)

        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
        # get only the external countours based on the hierarchy
        if hierarchy is not None:
            contours = [cnt for cnt, h in zip(contours, hierarchy[0]) if h[3] == -1]  # Keep only external contours

        bounding_boxes = [cv.boundingRect(cnt) for cnt in contours]

        nms_boxes_indices = cv.dnn.NMSBoxes(bounding_boxes, [conf] * len(bounding_boxes), class_confidence_threshold, iou_threshold, top_k=5)

        nms_boxes = bounding_boxes#[bounding_boxes[i] for i in nms_boxes_indices.flatten()] if nms_boxes_indices is not None else []


        color = get_class_color(class_id)

        for box in nms_boxes:
            (x, y, w, h) = box
            cv.rectangle(image, (x, y), (x + w, y + h), color, 4)
            cv.putText(image, f'ID: {get_class_name(class_id)}, Score: {conf:.2f}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            

       
        # Overlay mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        image = cv.addWeighted(image, 1.0, colored_mask, 0.1, 0)

    return image


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
        rgb_color = get_class_color(class_id)


        cv.rectangle(image, (x, y), (x + w, y + h), rgb_color, 2)
        # Optionally add text for class_id and score
        cv.putText(image, f'ID: {get_class_name(class_id)}, Score: {score:.4f}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, rgb_color, 1)


    return image  # Return the processed image



def inpaint_yolo_results_yolo11(results):
    image = results[-1]  # The last item is the image
    image = cv.resize(image, (640, 640))  # Resize the image to the target size if needed
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




class VideoPlaybackState(enum.Enum):
    PAUSED = 1
    PLAYING = 2
    SINGLE_FRAME = 3
    EXPORTING_RESULTS = 4
    STOPPED = 5

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
        self.frame_number = 0

        self.capture = cv.VideoCapture(self.video_source)

        self.state = VideoPlaybackState.STOPPED
        
        if self.capture.isOpened():
            self.valid_video_stream = True
            self.fps = self.capture.get(cv.CAP_PROP_FPS)
           


    def load_model(self, model_type):
        if self.running:
            self.running = False

        self.model_type = string_to_model_type(model_type)
        self.model = ort.InferenceSession(self.model_type.value, providers=[self.device])
        self.input_name = self.model.get_inputs()[0].name  # Get the input name of the model

        self.running = True

    def run(self):
        self.running = True
        self.state = VideoPlaybackState.PLAYING
        prev_time = time.time()

        while self.running and self.state != VideoPlaybackState.STOPPED:
            
            if not self.paused and self.state != VideoPlaybackState.PAUSED:
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

                    # print("output_image shape: ", output_image.shape)  # Debugging line
                   
                    # Run the model inference
                    try:

                        if self.model_type == ModelType.YOLO_8_D:
                            # extract the output from the model
                            output = self.model.run(None, {self.input_name: preprocessed_frame})[0]
                            output = np.squeeze(output)  # Remove batch dimension
                            
                            # process the output
                            yolo_results = postprocess_detections_yolo8(output, class_confidence_threshold=self.confidence_threshold,iou_threshold=self.iou_threshold)
                            # inpaint the detections on the output image
                            output_image = inpant_yolo8_detections(output_image, yolo_results)

                        elif self.model_type == ModelType.YOLOY_11_S:

                            output = self.model.run(None, {self.input_name: preprocessed_frame})

                            # print("output: ", output)  # Debugging line
                           
                            preds, protos = output  # Assuming the model returns two outputs


                            output_image = postprocess_detections_yolo11(output_image, preds, protos, class_confidence_threshold=self.confidence_threshold,
                                                            iou_threshold=self.iou_threshold)


                        self.prev = output_image.copy()  # Store the output image for paused state

                        qt_image = cv_image_to_qlabel(output_image)
                        self.frame_number += 1

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
                print("type(self.prev): ", type(self.prev))  # Debugging line

                output_image = cv_image_to_qlabel(self.prev)
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
        if self.running:
            self.single_frame_mode = enable

            if enable:
                self.state = VideoPlaybackState.SINGLE_FRAME
                

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