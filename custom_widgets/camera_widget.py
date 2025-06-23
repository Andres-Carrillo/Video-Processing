from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QWidget,QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from custom_workers.camera_worker import CameraWorker
from utils import qlabel_to_cv_image
from custom_widgets.camera_feed_editor import CameraFeedDialog
import cv2 as cv
class CameraWidget(QWidget):
    running = False
    
    def __init__(self):
        super().__init__()
        
        self.image_label = QLabel()
        layout = QVBoxLayout()
        blank_canvas = QPixmap(640, 480)
        blank_canvas.fill(Qt.black)
        self.image_label.setGeometry(0, 0, 640, 480)
        self.image_label.setScaledContents(True)
        self.image_label.setPixmap(blank_canvas)
        
        self.play_button = QPushButton("Start",clicked=self.start)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.image_label)
        layout.addWidget(self.play_button)
        self.setLayout(layout)

        self.camera_worker = CameraWorker()
        self.camera_worker.image.connect(self.update_image)

    def start(self):
        self.camera_worker.start()
        CameraWidget.running = True

        self.play_button.setText("Stop")
        self.play_button.clicked.disconnect()
        self.play_button.clicked.connect(self.closeEvent)    


    def set_video_source(self, video_source):
        if self.camera_worker.running:
            self.camera_worker.running = False
            self.camera_worker.wait()  # Wait for the thread to finish before changing the video source
            
        self.camera_worker.capture.release()  # Release the previous capture
        self.camera_worker.capture = cv.VideoCapture(video_source)  # Start a new capture with
        self.camera_worker.running = True
        self.camera_worker.start()  # Restart the thread with the new video source

    def update_video_source(self):

         # create widget that displays available cameras
        camera_feed_dialog = CameraFeedDialog(self)
        
        camera_feed_dialog.exec_()
        
        selected_camera = camera_feed_dialog.feed_option_widget.camera_index
        
        # if a camera is selected, set the camera source in the camera widget
        if selected_camera != -1:
            if selected_camera != self.camera_worker.camera_index:
                self.camera_worker.set_camera_index(selected_camera)


    def update_image(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        if CameraWidget.running:
            qlabel_to_cv_image(self.image_label)

    def closeEvent(self, event):
        self.camera_worker.stop()
        event.accept()
        CameraWidget.running = False