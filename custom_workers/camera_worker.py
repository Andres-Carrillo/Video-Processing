from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal
from utils import cv_image_to_qlabel
import cv2 as cv

class CameraWorker(QThread):
    image = pyqtSignal(QImage)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False

    def run(self):
        self.running = True
        capture = cv.VideoCapture(self.camera_index)
        while self.running:
            ret, frame = capture.read()

            if ret:
                # rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                qt_image = cv_image_to_qlabel(frame)
                self.image.emit(qt_image)
            
            else:
                break
        capture.release()

    def stop(self):
        self.running = False
        self.wait()