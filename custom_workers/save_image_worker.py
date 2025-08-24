from PyQt5.QtCore import  QThread, pyqtSignal,pyqtSlot
from PyQt5.QtGui import QImage
import cv2 as cv
from utils import qimage_to_cv_image
import queue

class ImageBlob():
    def __init__(self, image,save_path=''):
        self.image = image
        self.save_path = save_path

class YoloOutputBlob(ImageBlob):
    def __init__(self, image, detections, save_path=''):
        super().__init__(image, save_path)
        self.detections = detections

class SaveImageWorker(QThread):
    saved = pyqtSignal()
    empty_queue = pyqtSignal()
    image_list = queue.Queue()
    stop = False

    def __init__(self,save_path):
        super().__init__()
        self.save_path = save_path
    
    def run(self):

        while not self.stop and len(self.image_list) > 0:
            image_blob = self.image_list.get()
            self.save(image_blob[0],image_blob[1])
            self.image_list.task_done()
        
        self.saved.emit()

    def save(self,image,save_path):
        cv.imwrite(save_path,image)

    @pyqtSlot(QImage,str)
    def add_to_save_queue(self,image,file_name):
            cv_image = qimage_to_cv_image(image)

            full_path = self.save_path + "/" + file_name

            self.image_list.put(ImageBlob(cv_image,full_path)) 

    @pyqtSlot(str)
    def set_save_path(self,save_path): 
        self.save_path = save_path

    def stop_thread(self):
        self.data = None
        self.stop = True

        self.clear_queue()
        self.wait()


    def clear_queue(self):
        while not self.image_list.empty():
            self.image_list.get()
        
        self.image_list.task_done()

class SaveYoloOutPutWorker(SaveImageWorker):
    def __init__(self,save_path):
        super().__init__(save_path)
    
    def save(self,image,detections,save_path):
        # Convert QImage to OpenCV format
        cv_image = qimage_to_cv_image(image)

        print(f"Saving image to {save_path} with {len(detections)} detections.")

        # Save the image using OpenCV
        cv.imwrite(save_path, cv_image)

        # add saved txt file with yolo format


    def run(self):
        while not self.stop:
            if not self.image_list.empty():
                image_blob = self.image_list.get()
                self.save(image_blob.image,image_blob.detections,image_blob.save_path)
                self.image_list.task_done()
            else:
                self.msleep(100)


    @pyqtSlot(QImage,list,str)
    def add_to_save_queue(self,image,detections,file_name):
            cv_image = qimage_to_cv_image(image)

            full_path = self.save_path + "/" + file_name

            self.image_list.put(YoloOutputBlob(cv_image,detections,full_path))



