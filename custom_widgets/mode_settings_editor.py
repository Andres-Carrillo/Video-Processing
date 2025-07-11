# QDialgo to edit model settings
from PyQt5.QtWidgets import QWidget, QSizePolicy,QVBoxLayout, QLabel, QComboBox, QPushButton, QDialog,QSlider
from PyQt5.QtCore import Qt

class ModeSettingsEditor(QDialog):
    def __init__(self,parent=None,iou_threshold=0.5, confidence_threshold=0.5):
        super().__init__(parent)

        self.setWindowTitle("Model Settings Editor")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.layout = QVBoxLayout(self)

        self.mode_selector = QComboBox(self)
        self.mode_selector.addItems([mode.name.capitalize() for mode in ModelType])
        self.layout.addWidget(QLabel("Select Model Mode:"))
        self.layout.addWidget(self.mode_selector)

        self.confidence_threshold_input = QSlider(Qt.Horizontal, self)
        self.confidence_threshold_input.setRange(0, 100)  # Range from 0 to 100
        self.confidence_threshold_input.setValue(int(confidence_threshold * 100))
        self.confidence_threshold_input.setSingleStep(1)
        self.confidence_threshold_label = QLabel("Confidence Threshold:" + str(self.confidence_threshold_input.value()) + "%")

        self.layout.addWidget(self.confidence_threshold_label)
        self.layout.addWidget(self.confidence_threshold_input)

        self.iou_threshold_input = QSlider(Qt.Horizontal, self)
        self.iou_threshold_input.setRange(0, 100)  # Range from
        self.iou_threshold_input.setValue(int(iou_threshold * 100))
        self.iou_threshold_input.setSingleStep(1)
        self.iou_threshold_label = QLabel("IOU Threshold:" +str(self.iou_threshold_input.value()) + "%")

        self.layout.addWidget(self.iou_threshold_label)
        self.layout.addWidget(self.iou_threshold_input)

        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

        self.setLayout(self.layout)

        self.confidence_threshold_input.valueChanged.connect(self.update_confidence_threshold_label)
        self.iou_threshold_input.valueChanged.connect(self.update_iou_threshold_label)

    def update_confidence_threshold_label(self, value):
        self.confidence_threshold_label.setText("Confidence Threshold: " + str(value) + "%")

    def update_iou_threshold_label(self, value):
        self.iou_threshold_label.setText("IOU Threshold: " + str(value) +"%")

    def get_selected_mode(self):
        return self.mode_selector.currentText()
    
    def get_confidence_threshold(self):
        return self.confidence_threshold_input.value() / 100.0
    
    def get_iou_threshold(self):
        return self.iou_threshold_input.value() / 100.0
    

from custom_workers.onnx_video_worker import ModelType
class ModeSelectorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Select Model Mode")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.layout = QVBoxLayout(self)
        self.mode_selector = QComboBox(self)
        self.mode_selector.addItems([mode.name.capitalize() for mode in ModelType])
        self.layout.addWidget(QLabel("Select Model Mode:"))
        self.layout.addWidget(self.mode_selector)

        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

        self.setLayout(self.layout)

    def get_selected_mode(self):
        return self.mode_selector.currentText()