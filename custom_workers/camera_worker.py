from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal
from utils import cv_image_to_qlabel
import cv2 as cv
import time

class CameraWorker(QThread):
    image = pyqtSignal(QImage)

    def __init__(self, camera_index=-1,is_live_feed=True,video_path=None,fps=30):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.paused = False
        self.capture = None
        self.live_feed = is_live_feed  # Flag to indicate if the camera feed is live
        self.fps = fps  # Default FPS for the camera feed

        if not is_live_feed and video_path is not None:
            self.capture = cv.VideoCapture(video_path)
        else:
            self.capture = cv.VideoCapture(self.camera_index)
            self.fps = self.capture.get(cv.CAP_PROP_FPS)
    
    def run(self):
        self.running = True
        
        while self.running:
            frame_start = time.time()
            if not self.paused:
                ret, frame = self.capture.read()

                if ret:
                # rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    self.prev = qt_image = cv_image_to_qlabel(frame)
                    end_time = time.time()
                    elapsed_time = end_time - frame_start
                    sleep_time = max(0, (1 / self.fps) - elapsed_time)
                    time.sleep(sleep_time)  # Sleep to maintain the desired FPS
                    # Emit the processed frame
                    self.image.emit(qt_image)
            
                else:
                    break
            else:
                end_time = time.time()
                elapsed_time = end_time - frame_start
                sleep_time = max(0, (1 / self.fps) - elapsed_time)
                time.sleep(sleep_time)  # Sleep to maintain the desired FPS
                self.image.emit(self.prev)  # Emit the last frame while paused
        
        self.capture.release()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def set_video_path(self, video_path):
        if self.running:
            self.running = False
            self.wait()  # Wait for the thread to finish before changing the video path
        
        self.capture.release()
        self.capture = cv.VideoCapture(video_path)  # Start a new capture with the new
        self.fps = self.capture.get(cv.CAP_PROP_FPS)
        
        self.running = True
        self.start()

    def set_camera_index(self, camera_index):
        if self.running:
            self.running = False
            self.wait()  # Wait for the thread to finish before changing the camera index
        self.camera_index = camera_index

        self.capture.release()  # Release the previous capture
        self.capture = cv.VideoCapture(self.camera_index)  # Start a new capture with the new camera index
        
        self.running = True
        
        self.start()  # Restart the thread with the new camera index

    def stop(self):
        self.running = False
        self.wait()