from custom_widgets.camera_widget import CameraWidget
from custom_widgets.output_widget import OutputWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget,QHBoxLayout,QAction,QFileDialog
from custom_widgets.camera_feed_editor import CameraFeedDialog
import sys


## TODO: Connect self._camera_source to the camera widget so that it can be updated when the user selects a new camera feed

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera and Output Widget")

        
        file_menu = self.menuBar().addMenu("File")
        edit_menu = self.menuBar().addMenu("Edit")

        self.load_video_action = QAction("Load Video", self)
        self.set_live_camera_action = QAction("Change Live Source", self)

        self.change_upscale_model_action = QAction("Change Upscale Model", self)

        file_menu.addAction(self.load_video_action)
        file_menu.addAction(self.set_live_camera_action)

        edit_menu.addAction(self.change_upscale_model_action)
        self.camera_widget = CameraWidget()
        self.output_widget = OutputWidget(self)
        
        layout = QHBoxLayout()
        layout.addWidget(self.camera_widget)
        layout.addWidget(self.output_widget)
        
        container = QWidget()
        container.setLayout(layout)
        
        self.setCentralWidget(container)


        self.load_video_action.triggered.connect(self.update_video_source)
        self.set_live_camera_action.triggered.connect(self.update_camera_source)
        self.change_upscale_model_action.triggered.connect(self.update_upscale_model)

        self._camera_source = -1  # Default value for camera source


    def update_video_source(self):
        # create file dialog to select video file
        video_file, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_file:
            self.camera_widget.set_video_source(video_file)

    def update_camera_source(self):
        self.camera_widget.update_video_source()  # Stop the current video source if any


    def update_upscale_model(self):
        pass
        

if __name__ == "__main__":

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())