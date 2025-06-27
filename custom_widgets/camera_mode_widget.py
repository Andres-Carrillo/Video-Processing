from PyQt5.QtWidgets import QWidget,QComboBox,QVBoxLayout,QDialog, QPushButton
from PyQt5.QtWidgets import QSizePolicy


class CameraModeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Mode Selector")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.layout = QVBoxLayout(self)
        self.mode_selector = QComboBox(self)
        self.mode_selector.addItems(["Live Feed", "Snapshot"])
        self.layout.addWidget(self.mode_selector)

        self.setLayout(self.layout)

    def get_selected_mode(self):
        return self.mode_selector.currentText()
    

class CameraModeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Mode Selector")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.layout = QVBoxLayout(self)
        self.mode_selector = QComboBox(self)
        self.mode_selector.addItems(["Live Feed", "Snapshot"])
        self.layout.addWidget(self.mode_selector)

        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

        self.setLayout(self.layout)

    def get_selected_mode(self):
        return self.mode_selector.currentIndex()