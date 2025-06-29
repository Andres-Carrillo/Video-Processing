from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QWidget,QLabel,QHBoxLayout
from PyQt5.QtGui import QPixmap,QIcon
from PyQt5.QtCore import Qt
from custom_workers.camera_worker import CameraWorker, CameraFeedMode
from utils import qlabel_to_cv_image
from custom_widgets.camera_feed_editor import CameraFeedDialog
from custom_workers.camera_worker import VideoQueueWorker
from custom_workers.save_video_worker import SaveVideoWorker
from custom_workers.save_image_worker import SaveImageWorker
from custom_widgets.camera_mode_widget import CameraModeDialog
import cv2 as cv
import enum

import os


class CameraWidget(QWidget):
    running = False
    
    def __init__(self,parent=None, mode=CameraFeedMode.LIVE_FEED):
        super().__init__(parent)

        self.video_writer = None
        self.image_writer = None
        self.play_button = None
        self.record_button = None
        self.option_button = None
        self.forward_button = None
        self.backward_button = None
        self.image_label = None

        self.mode = mode
        self._init_ui()
   
        self.camera_worker = CameraWorker()
        self.camera_worker.image.connect(self.update_image)
         

    def _init_ui(self):
        self.image_label = QLabel()
        blank_canvas = QPixmap(640, 480)
        blank_canvas.fill(Qt.black)
        self.image_label.setGeometry(0, 0, 640, 480)
        self.image_label.setScaledContents(True)
        self.image_label.setPixmap(blank_canvas)

        layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        self.backward_button = QPushButton("", clicked=self.backward_frame)
        self.backward_button.setIcon(QIcon("../icons/back.svg"))
        self.backward_button.setToolTip("Previous Frame")
        button_layout.addWidget(self.backward_button)

        self.play_button = QPushButton("",clicked=self.start)
        self.play_button.setIcon(QIcon("../icons/play-svgrepo-com.svg"))
        self.play_button.setToolTip("Start Video Feed")
        button_layout.addWidget(self.play_button)

        self.image_label.setAlignment(Qt.AlignCenter)

        self.forward_button = QPushButton("", clicked=self.forward_frame)
        self.forward_button.setIcon(QIcon("../icons/forward-fill-svgrepo-com.svg"))
        self.forward_button.setToolTip("Next Frame")

        button_layout.addWidget(self.forward_button)

        self.record_button = QPushButton("", clicked=self.start_recording)
        self.record_button.setIcon(QIcon("../icons/not_recording-filled-alt-svgrepo-com.svg"))
        self.record_button.setToolTip("Start Recording")
        button_layout.addWidget(self.record_button)

        self.option_button = QPushButton("", clicked=self.display_options)
        self.option_button.setIcon(QIcon("../icons/options-svgrepo-com.svg"))
        self.option_button.setMaximumSize(30, 25)
        button_layout.addWidget(self.option_button)

        layout.addWidget(self.image_label)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.forward_button.setVisible(False)  # Initially hide the forward button
        self.backward_button.setVisible(False)  # Initially hide the backward button


    def swap_ui(self,mode):
        if mode == CameraFeedMode.LIVE_FEED or mode == CameraFeedMode.REAL_TIME_PROCESSING:
            self.play_button.setVisible(True)
            self.forward_button.setVisible(False)
            self.backward_button.setVisible(False)
            self.camera_worker.toggle_single_frame_mode(False)  # Disable single frame mode in the worker

        elif mode == CameraFeedMode.SINGLE_FRAME_CAPTURE:
            self.play_button.setVisible(False)
            self.forward_button.setVisible(True)
            self.backward_button.setVisible(True)
            self.camera_worker.toggle_single_frame_mode(True)  # Toggle single frame mode in the worker

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

    def display_options(self):
        camera_mode_dialog = CameraModeDialog(self)
        camera_mode_dialog.exec_()

        mode = camera_mode_dialog.get_selected_mode()
        mode = CameraFeedMode(mode) if isinstance(mode, int) else mode

        if mode != self.mode:
            self.swap_ui(mode)

            self.mode = mode

    def start_recording(self):
        self.record_button.setIcon(QIcon("../icons/recording-filled-svgrepo-com.svg"))
        self.record_button.setToolTip("Stop Recording")
        self.record_button.clicked.disconnect()
        self.record_button.clicked.connect(self.stop_recording)
        self.camera_worker.recording = True
        self.video_writer = SaveVideoWorker(save_path=os.path.join("Video-Super-Resolution/recordings/videos", "recorded_video.mp4"),
                                             fps=30,
                                             shape=(640, 480))
        self.video_writer.start()

    def stop_recording(self):
        self.record_button.setIcon(QIcon("../icons/not_recording-filled-alt-svgrepo-com.svg"))
        self.record_button.setToolTip("Stop Recording")
        self.record_button.clicked.disconnect()
        self.record_button.clicked.connect(self.start_recording)

    def start_image_recording(self):
        self.image_writer = SaveImageWorker(save_path=os.path.join("Video-Super-Resolution/recordings/images"))
        self.image_writer.start()

    def update_image(self, q_image):
     

        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

        if self.video_writer:
            frame = qlabel_to_cv_image(self.image_label)
            self.video_writer.add_to_queue(frame)

        if self.image_writer:
            frame = qlabel_to_cv_image(self.image_label)
            self.image_writer.add_to_queue(frame, f"frame{self.image_writer.frame_count}.jpg")

        if self.camera_worker.single_frame_mode and not self.camera_worker.paused:
            self.camera_worker.pause()

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
        self.play_button.setIcon(QIcon("../icons/pause-1006-svgrepo-com.svg"))
        self.play_button.setToolTip("Pause Video Feed")
        self.play_button.clicked.disconnect()
        self.play_button.clicked.connect(self.pause)

    def _set_resume_icon(self):
        self.play_button.setIcon(QIcon("../icons/play-svgrepo-com.svg"))
        self.play_button.setToolTip("Resume Video Feed")
        self.play_button.clicked.disconnect()
        self.play_button.clicked.connect(self.resume)

    def forward_frame(self):
        self.camera_worker.resume()

    def backward_frame(self):
        print("would call backward frame in camera worker")