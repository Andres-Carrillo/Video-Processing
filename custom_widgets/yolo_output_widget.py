from PyQt5.QtCore import Qt,pyqtSlot
from PyQt5.QtGui import  QPixmap
from PyQt5.QtWidgets import QWidget, QLabel,QGridLayout,QComboBox
from custom_workers.yolo_worker import YOLOWorker
import cv2 as cv
from utils import qimage_to_cv_image, cv_image_to_qimage,COCO_CLASSES, COCO_COLOR_LIST
import time

def get_class_name(class_id):
    return COCO_CLASSES.get(class_id, "Unknown")


def get_class_color(class_id):
    """
    Get the color for a given class ID.
    This function can be modified to return different colors based on the class ID.
    """    
    return COCO_COLOR_LIST.get(class_id, (255, 255, 255))  # Default to white if class_id not found


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
    processed_image = cv_image_to_qimage(image)

   
    
    return processed_image  # Return the processed image



# TODO: Add setting button to change model path, input size, confidence threshold, and IOU threshold, also model mode (detection, segmentation, etc.)
class YoloOutputWidget(QWidget):
    def __init__(self,parent,fixed_size=False,size=(640,600)):
        super().__init__(parent)
        self.setWindowTitle("YOLO Output Widget")
        
        self.canvas = QPixmap(size[0], size[1])
        self.label = QLabel(parent=self)
        
        self.canvas.fill(Qt.black)
        self.label.setPixmap(self.canvas)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)
        
        layout = QGridLayout()
        layout.addWidget(self.label, 0, 0)
           
        self.setLayout(layout)
        
        if fixed_size:
            self.setFixedSize(size[0], size[1])

        self._init_worker()

    def _init_worker(self):
        self.model_worker = YOLOWorker(model_path="model_zoo/yolov8n.onnx", input_size=640, conf_threshold=0.5, iou_threshold=0.4)
    
    def start_worker(self):
        self.model_worker.results.connect(self.update_canvas)
        self.model_worker.start()

    @pyqtSlot(list)
    def update_canvas(self, data):
            if not isinstance(data, list):
                #set the canvas to black if data is not a list
                self.canvas.fill(Qt.black)
            elif len(data) > 0:
                data = inpaint_yolo_results(data)
                data = QPixmap.fromImage(data) if not isinstance(data, QPixmap) else data

                self.label.setPixmap(data)
                self.label.repaint()
                print("returning processed image")

            else:
                print("No data received or empty list")
