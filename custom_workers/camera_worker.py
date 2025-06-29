from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal
from utils import cv_image_to_qlabel
import cv2 as cv
import time
import queue
import enum

class CameraFeedMode(enum.Enum):
    LIVE_FEED = 0
    SINGLE_FRAME_CAPTURE = 1
    REAL_TIME_PROCESSING = 2


class CameraWorker(QThread):
    image = pyqtSignal(QImage)

    def __init__(self, camera_index=-1,video_path=None,fps=30,limit_fps=True,camera_mode=CameraFeedMode.LIVE_FEED):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.paused = False
        self.capture = None
        self.limited_fps = limit_fps  # Flag to indicate if FPS limiting is enabled
        self.fps = fps  # Frames per second limit
        self.wait_for_next = False  # Flag to indicate if we are waiting for the next frame in single frame mode
        self.single_frame_mode = False
        self.prev = None  # To store the previous frame for paused state


        if camera_mode == CameraFeedMode.REAL_TIME_PROCESSING and video_path is not None:
            self.capture = cv.VideoCapture(video_path)
        else:
            self.capture = cv.VideoCapture(self.camera_index)
            if camera_mode == CameraFeedMode.SINGLE_FRAME_CAPTURE:
                self.single_frame_mode = True

    def run(self):
        self.running = True
        prev_time = time.time()

        while self.running:
            if not self.paused:
                # FPS limiting logic
                if self.limited_fps and self.fps > 0:
                    current_time = time.time()
                    elapsed = current_time - prev_time
                    wait_time = max(0, (1.0 / self.fps) - elapsed)
                    if wait_time > 0:
                        self.msleep(int(wait_time * 1000))
                    prev_time = time.time()

                ret, frame = self.capture.read()

                if ret:
                    self.prev = qt_image = cv_image_to_qlabel(frame)
                    self.image.emit(qt_image)
                else:
                    break
            else:
                self.image.emit(self.prev)
                self.msleep(100)

        self.capture.release()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


    def toggle_single_frame_mode(self, enable):
        print(f"Toggling single frame mode to {enable}")
        self.single_frame_mode = enable

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

##TODO: Rewrite thread safety to not reliy on qsize
class VideoQueueWorker(QThread):
    batch_ready = pyqtSignal(list)
    pause_processing = pyqtSignal()
    resume_processing = pyqtSignal()
    
    def __init__(self, batch_size=30,max_queue_size=100):
        super().__init__()
        self.queue = queue.Queue(maxsize=max_queue_size)  # Set a maximum size for the queue to avoid memory issues
        self.running = False
        self.batch_size = batch_size
        self.backlog = list()  # This will hold frames that were not placed in the queue due to it being full


    def enqueue_frame(self, frame):
        if not self.running:
            raise RuntimeError("Worker is not running. Start the worker before enqueuing frames.")

        try:
            if len(self.backlog) > 0:  # If there are frames in the backlog, we try to put them in the queue first
                backlog_frame = self.backlog.pop(0)
                self.queue.put(backlog_frame, timeout=1)  # Wait at most one second to put the frame in the queue
   
            else:
                # If the backlog is empty, we can add the new frame to the queue
                
                self.queue.put(frame, timeout=1)  # Wait at most one second to put the frame in the queue
                self.resume_processing.emit()  # Emit signal to resume processing if the queue was previously full
                
        except queue.Full:
            self.backlog.append(frame)  # If the queue is full, add the frame to the backlog
            print("Queue is full, adding frame to backlog.")
            self.pause_processing.emit()  # Emit signal to pause processing if the queue is full


    def run(self):
        self.running = True
        batch = list()

        while self.running:
            try:
                 # avoid blocking if the queue is empty
                frame = self.queue.get(timeout=1)  # Wait at most one second for a frame to be available
                batch.append(frame)
          
                if len(batch) == self.batch_size:
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