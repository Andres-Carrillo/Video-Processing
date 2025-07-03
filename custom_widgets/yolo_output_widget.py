from PyQt5.QtCore import Qt
from PyQt5.QtGui import  QPixmap
from PyQt5.QtWidgets import QWidget, QLabel,QGridLayout,QComboBox
from custom_workers.yolo_worker import YOLOWorker

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

    def _init_worker(self):
        self.worker = YOLOWorker(model_path="yolov8n.onnx", input_size=640, conf_threshold=0.5, iou_threshold=0.4)
        self.worker.results.connect(self.update_canvas)
        self.worker.start()

    def update_canvas(self, data):
        if data is not None:
            print("Updating canvas with new data")
            print(f"Data type: {type(data)}")
            #convert the data to QPixmap if it is not already
            if not isinstance(data, QPixmap):
                print("Converting data to QPixmap")
                data = QPixmap.fromImage(data)
            else:
                print("Data is already a QPixmap")
            self.canvas = data
            self.label.setPixmap(self.canvas)
            self.label.repaint()
