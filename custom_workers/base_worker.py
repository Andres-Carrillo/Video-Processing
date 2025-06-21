from PyQt5.QtCore import  QThread, pyqtSignal,pyqtSlot
from PyQt5.QtGui import QImage


class BaseImageWorker(QThread):
    processed = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.data = None

    @pyqtSlot(QImage)
    def process(self,data):

        self.process_data(data)
        
        if self.data is None:
            self.processed.emit(QImage())
        
        else:
            self.processed.emit(self.data)
        
        self.finished.emit()

    def process_data(self, data):
        self.data = data

    def stop_thread(self):
        self.data = None
        self.wait()