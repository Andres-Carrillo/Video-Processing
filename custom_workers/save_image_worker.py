from PyQt5.QtCore import  QThread, pyqtSignal,pyqtSlot
from PyQt5.QtGui import QImage
import cv2 as cv
from utils import qimage_to_cv_image,cv_image_to_qimage
import numpy as np

class ImageBlob():
    def __init__(self, image, save_path):
        self.image = image
        self.save_path = save_path

class SaveImageWorker(QThread):
    saved = pyqtSignal()
    empty_queue = pyqtSignal()
    image_list = []
    stop = False

    def __init__(self,save_path):
        super().__init__()
        self.save_path = save_path
        print("built save image worker")
    
    def run(self):

        while not self.stop and len(self.image_list) > 0:
            image_blob = self.image_list.pop(0)
            self.save(image_blob[0],image_blob[1])
        
        self.saved.emit()

    def save(self,image,save_path):
        cv.imwrite(save_path,image)

    def add_to_save_queue(self,image,file_name):
            cv_image = qimage_to_cv_image(image)

            full_path = self.save_path + "/" + file_name

            self.image_list.append([cv_image,full_path]) 

    @pyqtSlot(QImage,str)
    def add_to_queue(self,image,file_name):
        self.image_list.append(ImageBlob(image,file_name))
            
    @pyqtSlot(str)
    def set_save_path(self,save_path): 
        self.save_path = save_path
    
    def add_to_queue(self,image,file_name):
        self.image_list.append(ImageBlob(image,file_name))

    def stop_thread(self):
        self.data = None
        self.stop = True
        self.wait()



