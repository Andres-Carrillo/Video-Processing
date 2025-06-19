from custom_widgets.camera_widget import CameraWidget
from custom_widgets.output_widget import OutputWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget,QHBoxLayout

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera and Output Widget")
        
        self.camera_widget = CameraWidget()
        self.output_widget = OutputWidget(self)
        
        layout = QHBoxLayout()
        layout.addWidget(self.camera_widget)
        layout.addWidget(self.output_widget)
        
        container = QWidget()
        container.setLayout(layout)
        
        self.setCentralWidget(container)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())