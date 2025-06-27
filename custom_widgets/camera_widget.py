from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QWidget,QLabel,QHBoxLayout
from PyQt5.QtGui import QPixmap,QIcon
from PyQt5.QtCore import Qt
from custom_workers.camera_worker import CameraWorker
from utils import qlabel_to_cv_image
from custom_widgets.camera_feed_editor import CameraFeedDialog
from custom_workers.camera_worker import VideoQueueWorker
import cv2 as cv
import os
class CameraWidget(QWidget):
    running = False
    
    def __init__(self):
        super().__init__()
        
        self.image_label = QLabel()
        layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        blank_canvas = QPixmap(640, 480)
        blank_canvas.fill(Qt.black)
        self.image_label.setGeometry(0, 0, 640, 480)
        self.image_label.setScaledContents(True)
        self.image_label.setPixmap(blank_canvas)
        
        self.play_button = QPushButton("",clicked=self.start)
        icon_path = os.path.join(os.path.dirname(__file__), "../play-svgrepo-com.svg")
        self.play_button.setIcon(QIcon(icon_path))

        self.image_label.setAlignment(Qt.AlignCenter)

        self.option_button = QPushButton("", clicked=self.display_options)
        option_icon_path = os.path.join(os.path.dirname(__file__), "../options-svgrepo-com.svg")
        self.option_button.setIcon(QIcon(option_icon_path))
        self.option_button.setMaximumSize(30, 25)

        layout.addWidget(self.image_label)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.option_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.camera_worker = CameraWorker()
        self.camera_worker.image.connect(self.update_image)
        
        # self.frame_queue_worker = VideoQueueWorker()


    def start(self):
        self.camera_worker.start()
        # self.frame_queue_worker.start()

        CameraWidget.running = True

        self._set_pause_icon()


    def set_video_source(self, video_source):
        if self.camera_worker.running:
            self.camera_worker.running = False
            self.camera_worker.wait()  # Wait for the thread to finish before changing the video source
            
        self.camera_worker.capture.release()  # Release the previous capture
        self.camera_worker.capture = cv.VideoCapture(video_source)  # Start a new capture with
        self.camera_worker.running = True
        self.camera_worker.start()  # Restart the thread with the new video source


        self._set_pause_icon()

    def update_video_source(self):

         # create widget that displays available cameras
        camera_feed_dialog = CameraFeedDialog(self)
        
        camera_feed_dialog.exec_()
        
        selected_camera = camera_feed_dialog.feed_option_widget.camera_index
        
        # if a camera is selected, set the camera source in the camera widget
        if selected_camera != -1:
            if selected_camera != self.camera_worker.camera_index:
                self.camera_worker.set_camera_index(selected_camera)


    # TODO: Add functionality to change the feed mode.
    #      should allow for continuous feed, single frame capture controlled by user, so each frame can be processed individually
    #      and a mode that allows for processing of the video feed in real-time.
    #      The mode can be set in the options dialog.
    #      The options dialog should also allow for changing the camera source.
    def display_options(self):
        print("Displaying options dialog")


    def update_image(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        qlabel_to_cv_image(self.image_label)

    def closeEvent(self, event):
        self.camera_worker.stop()
        CameraWidget.running = False

    def pause(self):
        if self.camera_worker.running:
            self.camera_worker.pause()

            self._set_resume_icon()

    def resume(self):
        if not self.camera_worker.running:
            self.camera_worker.running = True
            self.camera_worker.start()
        else:
            self.camera_worker.resume()

        self._set_pause_icon()



    def _set_pause_icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), "../pause-1006-svgrepo-com.svg")
        self.play_button.setIcon(QIcon(icon_path))
        self.play_button.clicked.disconnect()
        self.play_button.clicked.connect(self.pause)


    def _set_resume_icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), "../play-svgrepo-com.svg")
        self.play_button.setIcon(QIcon(icon_path))
        self.play_button.clicked.disconnect()
        self.play_button.clicked.connect(self.resume)