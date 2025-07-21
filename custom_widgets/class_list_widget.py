from PyQt5.QtWidgets import (QWidget,  QPushButton,QColorDialog,QInputDialog,QVBoxLayout,QListWidget,QMessageBox,QHBoxLayout, QListWidgetItem,QLabel)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor
from random import randint
from utils import COCO_COLOR_LIST,COCO_CLASSES
from custom_workers.onnx_video_worker import ModelType

class ClassItem(QWidget):
    def __init__(self, text, color, parent=None,row_index=0):
        super().__init__(parent)
        self._class_name = text
        self._color = color
   
        # Handle both QColor and tuple
        if isinstance(color, QColor):
            r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()
        else:
            r, g, b = color[0], color[1], color[2]
            a = color[3] if len(color) > 3 else 255

        rgba_str = f"rgba({r}, {g}, {b}, {a})"

        layout = QHBoxLayout()
        layout.setSpacing(0)  # Remove spacing between widgets
        layout.setContentsMargins(0, 0, 0, 0)  #
        self.label = QLabel(text)
        self.label.setFixedSize(90, 20)

        if row_index%2 == 0:
            self.label.setStyleSheet(f"background-color: rgba(232, 232, 232,175);")

        self.edit_button = QPushButton("")

        self.edit_button.setFixedWidth(40)
        self.edit_button.setFixedHeight(20)
        self.edit_button.setStyleSheet(f"background-color: {rgba_str};")
 
        layout.addWidget(self.label)
        layout.addWidget(self.edit_button)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(0)
        self.setLayout(layout)

class ClassListWidget(QWidget):
    class_added = pyqtSignal(str, QColor)
    class_edited = pyqtSignal(str, QColor, int)
    class_removed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.add_class_button = QPushButton("Add Class")
        layout.addWidget(self.list_widget)
        layout.addWidget(self.add_class_button)
        self.setLayout(layout)

        self.add_class_button.clicked.connect(self.add_class)
        self.list_widget.itemClicked.connect(self.item_clicked)
        self.list_widget.itemDoubleClicked.connect(self.remove_item)

        self.setSizePolicy(
            QWidget.sizePolicy(self).horizontalPolicy(),
            QWidget.sizePolicy(self).verticalPolicy()
        )

        # self.setFixedSize(300, 400)  # Set a fixed size for the widget
        self.setFixedWidth(160)  # Set a fixed width for the widget

    def add_class(self):
        class_name, _ = QInputDialog.getText(self, "Class Name", "Enter Class Name")
        if not class_name:
            return
        init_color = QColor(randint(0,255), randint(0,255), randint(0,255), 175)
        item = QListWidgetItem()
        widget = ClassItem(class_name, init_color)
        item.setSizeHint(widget.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, widget)
        self.class_added.emit(class_name, init_color)

    def item_clicked(self, item):
        widget = self.list_widget.itemWidget(item)
        if widget:
            edit_class_name, _ = QInputDialog.getText(self, "Class Name", "Edit Class Name", text=widget._class_name)
            if edit_class_name:
                widget.label.setText(edit_class_name)
                widget._class_name = edit_class_name
            color = QColorDialog.getColor()
            if color.isValid():
                widget.edit_button.setStyleSheet(f"background-color: {color.name()};")
                widget._color = color
            self.class_edited.emit(widget._class_name, widget._color, self.list_widget.row(item))

    def remove_item(self, item):
        should_remove = QMessageBox(self)
        should_remove.setWindowTitle("Remove Class?")
        should_remove.setText("Are you sure you want to remove this class?")
        should_remove.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        button = should_remove.exec_()
        if button == QMessageBox.Yes:
            row = self.list_widget.row(item)
            self.list_widget.takeItem(row)
            self.class_removed.emit(row)

    def import_classes(self, model_type=ModelType.YOLO_8_D):
        classes = COCO_CLASSES if model_type == ModelType.YOLO_8_D or model_type == ModelType.YOLOY_11_S else []
        classes = [(COCO_CLASSES[class_id], COCO_COLOR_LIST[i % len(COCO_COLOR_LIST)]) for i, class_id in enumerate(classes)]
        self.list_widget.clear()
        for index, (class_id, color) in enumerate(classes):
            item = QListWidgetItem()
            widget = ClassItem(class_id, color, row_index=index)
            item.setSizeHint(widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)