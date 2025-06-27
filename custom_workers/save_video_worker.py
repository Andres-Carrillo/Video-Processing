from PyQt5.QtCore import  QThread, pyqtSignal,pyqtSlot
from PyQt5.QtGui import QImage
import cv2 as cv
from utils import qimage_to_cv_image,cv_image_to_qimage
import numpy as np

class SaveVideoWorker(QThread):
    saved = pyqtSignal()
    empty_queue = pyqtSignal()
    video_list = []
    stop = False

    def __init__(self,save_path,fps=30,shape=(640,480),pause=True,codec_codec='mp4v'):
        super().__init__()
        self.save_path = save_path
        self.fps = fps
        self.pause = pause
        self.running = False
        fourcc = cv.VideoWriter_fourcc(*codec_codec)
        height, width, _ = shape
        self.writer =  cv.VideoWriter(save_path, fourcc, self.fps, (width, height))

    def run(self):
        self.running = True
        # while the thread is running, keep writing frames to the video file
        while self.running:
            # if the thread is paused, just wait
            if not self.pause:
                # if the queue is empty, emit the empty_queue signal
                # otherwise pop a frame from the queue and write it to the video file
                if len(self.video_list) > 0:
                    frame = self.video_list.pop(0)
                    self.writer.write(frame)
                else:
                    self.empty_queue.emit()

        self.saved.emit()

    def start(self):
        self.running = True
        self.video_list = []

    def stop_recording(self):
        self.running = False

        # clear out the queue and save the remaining frames
        while len(self.video_list) > 0:
            frame = self.video_list.pop(0)
            self.writer.write(frame)

        self.saved.emit()
        self.writer.release()
        self.wait()

    @pyqtSlot(QImage)
    def add_to_queue(self,image):
        cv_image = qimage_to_cv_image(image)
        self.video_list.append(cv_image)