# QDialgo to edit model settings
from PyQt5.QtWidgets import QWidget, QSizePolicy,QVBoxLayout, QLabel, QGridLayout, QComboBox, QPushButton, QDialog, QLineEdit
from PyQt5.QtCore import Qt



class ModeSettingsEditor(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)

        self.setWindowTitle("Model Settings Editor")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.layout = QVBoxLayout(self)
        self.mode_selector = QComboBox(self)
        self.mode_selector.addItems(["Detection", "Segmentation", "Classification"])
        self.layout.addWidget(QLabel("Select Model Mode:"))
        self.layout.addWidget(self.mode_selector)

        self.confidence_threshold_label = QLabel("Confidence Threshold:")
        self.confidence_threshold_input = QLineEdit(self)
        self.confidence_threshold_input.setPlaceholderText("Enter confidence threshold (0.0 - 1.0)")
        self.layout.addWidget(self.confidence_threshold_label)
        self.layout.addWidget(self.confidence_threshold_input)

        self.iou_threshold_label = QLabel("IOU Threshold:")
        self.iou_threshold_input = QLineEdit(self)
        self.iou_threshold_input.setPlaceholderText("Enter IOU threshold (0.0 - 1.0)")
        self.layout.addWidget(self.iou_threshold_label)
        self.layout.addWidget(self.iou_threshold_input)

        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)


        self.setLayout(self.layout)

    def get_selected_mode(self):
        return self.mode_selector.currentText()