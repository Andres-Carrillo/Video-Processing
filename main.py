from custom_widgets.camera_widget import CameraWidget
from custom_widgets.output_widget import OutputWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget,QHBoxLayout,QAction,QFileDialog
from custom_widgets.camera_feed_editor import CameraFeedDialog
from custom_widgets.yolo_output_widget import YoloOutputWidget
import sys

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
        # self.output_widget = OutputWidget(self)
        # self.yolo_output_widget = YoloOutputWidget(self)

        layout = QHBoxLayout()
        layout.addWidget(self.camera_widget)
        # layout.addWidget(self.output_widget)
        # layout.addWidget(self.yolo_output_widget)

        container = QWidget()
        container.setLayout(layout)
        
        self.setCentralWidget(container)

        # self.camera_widget.camera_worker.image.connect(self.output_widget.worker.process)
        # self.camera_widget.camera_worker.image.connect(self.yolo_output_widget.model_worker.processing_list.put)


        self.load_video_action.triggered.connect(self.update_video_source)
        self.set_live_camera_action.triggered.connect(self.update_camera_source)
        self.change_upscale_model_action.triggered.connect(self.update_upscale_model)

        self._camera_source = -1  # Default value for camera source

        # self.yolo_output_widget.start_worker()  # Start the YOLO worker thread


    def update_video_source(self):
        # create file dialog to select video file
        video_file, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_file:
            self.camera_widget.set_video_source(video_file)

    def update_camera_source(self):
        self.camera_widget.update_video_source()  # Stop the current video source if any


    def update_upscale_model(self):
        pass


    # def closeEvent(self, event):
    #     self.camera_widget.camera_worker.running = False
    #     self.camera_widget.camera_worker.wait()  # Wait for the camera worker thread to finish
    #     self.camera_widget.camera_worker.stop()  # Stop the camera worker thread

    #     self.output_widget.worker.wait()  # Wait for the output worker thread to finish
    #     self.output_widget.worker.stop_thread()
    #     # Signal all threads to stop
    #     # # self.stop_event.set()
    #     # for worker in self.workers:
    #     #     worker.join()  # Wait for threads to finish
    #     # event.accept()
        

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