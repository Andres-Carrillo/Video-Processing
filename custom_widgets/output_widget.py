from PyQt5.QtCore import Qt
from PyQt5.QtGui import  QPixmap
from PyQt5.QtWidgets import QWidget, QLabel,QGridLayout,QComboBox
from custom_workers.upscale_worker import UpscaleWorker

class OutputWidget(QWidget):
    def __init__(self,parent,fixed_size=False,size=(640, 600)):
        super().__init__(parent)
        self.saving_masks = False

        self._init_ui(fixed_size, size)
        self._init_worker()

    def _init_ui(self,is_fixed_size, size):
        self.setWindowTitle("Output Widget")
        
        self.canvas = QPixmap(size[0], size[1])
        self.label = QLabel(parent=self)

        self.canvas.fill(Qt.black)
        self.label.setPixmap(self.canvas)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)
        self.scale_factor = QComboBox(self)

        self.scale_factor.addItem("2x Upscale")
        self.scale_factor.addItem("4x Upscale")

        layout = QGridLayout()
        layout.addWidget(self.label, 0, 0)
        layout.addWidget(self.scale_factor, 1, 0)

        self.setLayout(layout)

        if is_fixed_size:
            self.setFixedSize(size[0], size[1])

    
    def _init_worker(self):
        self.worker = UpscaleWorker(model_type="esrgan", is_deep_learning=True)
        self.worker.processed.connect(self.update_canvas)
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

            