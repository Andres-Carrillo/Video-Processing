from PyQt5.QtWidgets import QWidget,QFileDialog,QComboBox,QVBoxLayout,QDialog
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy
import cv2 as cv
from cv2_enumerate_cameras import enumerate_cameras 

class CameraFeedWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Camera Feed")
        self._init_camera_list()

        # given the camera_list create a combo box to select the camera
        self.feed_option_widget = QComboBox(self)

        for camera_index in self.camera_list:
            self.feed_option_widget.addItem(f"Camera {camera_index}", camera_index)

        layout = QVBoxLayout()

        layout.addWidget(self.feed_option_widget)
        self.setLayout(layout)

        self.feed_option_widget.currentIndexChanged.connect(self._on_camera_selected)
    
    def _on_camera_selected(self, index):
        print(f"Selected camera index: {index}")
        self.camera_index = index

    def _init_camera_list(self):
        found = set()
        self.camera_list = []
        camera_info = enumerate_cameras()
        self.camera_index = -1

        for camera in camera_info:    
            if camera.pid in found:
                continue
            found.add(camera.pid)
            self.camera_list.append(camera.name)

#         if no cameras are found, add a dummy camera
        if len(self.camera_list) < 1:
            self.camera_list.append("No cameras found")
        else:
            self.camera_index = 0

# dialog to wrap the camera feed widget
class CameraFeedDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Camera Feed")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.feed_option_widget = CameraFeedWidget(self)
        self.feed_option_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout()
        layout.addWidget(self.feed_option_widget)
        self.setLayout(layout)

        self.setModal(True)