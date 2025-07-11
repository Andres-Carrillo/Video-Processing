from custom_widgets.camera_widget import CameraWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget,QHBoxLayout,QAction,QFileDialog
from custom_widgets.mode_settings_editor import ModeSettingsEditor
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Camera and Output Widget")

        file_menu = self.menuBar().addMenu("File")
        edit_menu = self.menuBar().addMenu("Edit")

        self.load_video_action = QAction("Load Video", self)
        self.set_live_camera_action = QAction("Change Live Source", self)

        self.edit_model_settings_action = QAction("Edit Model Settings", self)

        file_menu.addAction(self.load_video_action)
        file_menu.addAction(self.set_live_camera_action)

        # edit_menu.addAction(self.change_model_action)
        edit_menu.addAction(self.edit_model_settings_action)
 
        self.camera_widget = CameraWidget()

        layout = QHBoxLayout()
        layout.addWidget(self.camera_widget)

        container = QWidget()
        container.setLayout(layout)
        
        self.setCentralWidget(container)

        self.load_video_action.triggered.connect(self.update_video_source)
        self.set_live_camera_action.triggered.connect(self.update_camera_source)
        self.edit_model_settings_action.triggered.connect(self.edit_model_settings)
       
        self._camera_source = -1  # Default value for camera source

    def update_video_source(self):
        # create file dialog to select video file
        video_file, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_file:
            self.camera_widget.set_video_source(video_file)

    def update_camera_source(self):
        self.camera_widget.update_video_source()  # Stop the current video source if any

    def _load_new_model(self, selected_model):
        if selected_model!= self.camera_widget.camera_worker.model_type.name:
            self.camera_widget.camera_worker.load_model(selected_model)

    def edit_model_settings(self):
        # Open a dialog to edit model settings
        dialog = ModeSettingsEditor(self,self.camera_widget.camera_worker.iou_threshold, self.camera_widget.camera_worker.confidence_threshold)
        dialog.exec_()
        selected_model = dialog.mode_selector.currentText().upper()
        self._load_new_model(selected_model)

        confidence_threshold = dialog.get_confidence_threshold()
        iou_threshold = dialog.get_iou_threshold()
        self.camera_widget.camera_worker.set_confidence_threshold(confidence_threshold)
        self.camera_widget.camera_worker.set_iou_threshold(iou_threshold)

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"An error occurred: {e}")
        # main_window.output_widget.worker.video_writer.release()  # Ensure video writer is released
        sys.exit(1)