from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal
from utils import cv_image_to_qlabel
import cv2 as cv
import time
import queue

# TODO: batch processing of frames
#       # wait the needed time to maintain the desired FPS
#       # However, continue

class CameraWorker(QThread):
    image = pyqtSignal(QImage)

    def __init__(self, camera_index=-1,is_live_feed=True,video_path=None):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.paused = False
        self.capture = None
        self.live_feed = is_live_feed  # Flag to indicate if the camera feed is live

        if not is_live_feed and video_path is not None:
            self.capture = cv.VideoCapture(video_path)
        else:
            self.capture = cv.VideoCapture(self.camera_index)
    
    def run(self):
        self.running = True
        
        while self.running:
            if not self.paused:
                ret, frame = self.capture.read()

                if ret:
                # rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    self.prev = qt_image = cv_image_to_qlabel(frame)
                    # Emit the processed frame
                    self.image.emit(qt_image)
            
                else:
                    break
            else:

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

# this worker is used to provide a queue of frames from a video file
# it can be used to process frames in batches or to provide a live feed from a video
# it emits frames as QImage objects
# it must take in frames from another thread (e.g. CameraWorker)
# and emit them to another thread wich processes them (e.g. VideoProcessingWorker)
class VideoQueueWorker(QThread):
    batch_ready = pyqtSignal(list)
    pause_processing = pyqtSignal()
    resume_processing = pyqtSignal()
    
    def __init__(self, batch_size=1,max_queue_size=100):
        super().__init__()
        self.queue = queue.Queue(maxsize=max_queue_size)  # Set a maximum size for the queue to avoid memory issues
        self.running = False
        self.batch_size = batch_size


    def enqueue_frame(self, frame):
        if not self.running:
            raise RuntimeError("Worker is not running. Start the worker before enqueuing frames.")

        if len(self.queue) == 1 - self.queue.maxsize:
            self.pause_processing.emit()  # Emit signal to pause processing if the queue is about to be full

        else: # if there is space in the queue, we add the frame to it
            self.queue.put(frame)

            if self.queue.qsize() < self.queue.maxsize//2: # is the queue is less than half full, we resume processing
                self.resume_processing.emit()
        


    def run(self, batch_size):
        self.running = True
        batch = list()

        while self.running:
            try:
                if len(self.queue) > 0: # avoid blocking if the queue is empty
                    frame = self.queue.get(timeout=1)  # Wait at most one second for a frame to be available
                    batch.append(frame)

                    if len(batch) == batch_size:
                        self.batch_ready.emit(batch)
                        batch = list()  # Reset the batch after emitting

            except queue.Empty: # if no frame is available in the queue then we check if we have a batch to emit
                if batch:
                    if len(batch) > 0:
                        # Emit the remaining batch if it has frames
                        self.batch_ready.emit(batch)
                        batch = list()
                    else:
                        self.batch_ready.emit(None)  # Reset the batch after emitting


    def stop(self):
        self.running = False
        self.wait()