import sys
import cv2
import numpy as np
import logging
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QSettings, Qt
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QComboBox, QMainWindow, \
    QMessageBox, QAction, QSplashScreen
import os
import torch
from ultralytics import YOLO

# 禁用控制台输出日志
logging.getLogger().setLevel(logging.CRITICAL)


class FaceRecognitionWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV-RollCall")
        self.resize(800, 480)

        # 加载设置
        self.settings = QSettings("settings.ini", QSettings.IniFormat)
        self.detection_active = False  # 跟踪检测状态
        self.init_ui()  # 初始化界面

    def init_ui(self):
        # 添加菜单栏
        menubar = self.menuBar()
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.about_dialog)
        menubar.addAction(about_action)

        # 模型选择
        self.model_selector = QComboBox()
        self.model_selector.addItem("YOLOv8", "yolov8")
        self.model_selector.currentIndexChanged.connect(self.select_model)
        self.model_selector.setStyleSheet("background-color: #ffffff; border: 2px solid #4CAF50; border-radius: 12px;")

        self.face_model = None
        self.select_model(0)

        # 摄像头选择
        self.camera_combobox = QComboBox()
        self.populate_cameras()
        self.camera_combobox.currentIndexChanged.connect(self.select_camera)
        self.camera_combobox.setStyleSheet("background-color: #ffffff; border: 2px solid #4CAF50; border-radius: 12px;")

        # 打开摄像头
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 定时器用于更新帧
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 显示摄像头帧的标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        # 显示人脸数的标签
        self.face_count_label = QLabel("人脸数：0")

        # 开始和暂停按钮
        self.start_button = QPushButton("开始")
        self.start_button.setStyleSheet("background-color: #4CAF50; border: none; color: white; padding: 10px 24px;"
                                        "text-align: center; text-decoration: none; display: inline-block; font-size: 16px;"
                                        "margin: 4px 2px; cursor: pointer; border-radius: 12px;")
        self.start_button.clicked.connect(self.start_detection)

        self.pause_button = QPushButton("停止")
        self.pause_button.setStyleSheet("background-color: #FF5733; border: none; color: white; padding: 10px 24px;"
                                        "text-align: center; text-decoration: none; display: inline-block; font-size: 16px;"
                                        "margin: 4px 2px; cursor: pointer; border-radius: 12px;")
        self.pause_button.clicked.connect(self.pause_detection)

        # 设置布局
        layout = QVBoxLayout()
        layout.addWidget(self.image_label, 1)  # 1 表示伸缩因子为 1，使图像标签可以随窗口大小变化而等比例缩放
        layout.addWidget(self.face_count_label)

        # 添加伸缩项以填充空白空间
        layout.addStretch()

        layout.addWidget(self.camera_combobox)
        layout.addWidget(self.model_selector)
        layout.addWidget(self.start_button)
        layout.addWidget(self.pause_button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # 加载设置
        self.load_settings()

        # 初始状态下启用开始和停止按钮
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(True)

    def resizeEvent(self, event):
        # 在窗口大小变化时，重新调整图像标签的大小以保持比例
        super().resizeEvent(event)
        self.update_frame()

    def about_dialog(self):
        about_text = ("CV-RollCall\n\n版本: v0.3\n\nGitHub仓库\nhttps://github.com/Return-Log/CV-RollCall\n\n许可证: "
                      "AGPL-3.0\n\nCopyright © 2024 Log. All rights reserved.")
        QMessageBox.about(self, "关于", about_text)

    def select_model(self, index):
        model_type = self.model_selector.itemData(index)
        if model_type == "yolov8":
            self.face_model = YOLO("models/yolov8s-face.pt")

    def populate_cameras(self):
        camera_list = []
        index = 0
        while True:
            camera = cv2.VideoCapture(index)
            if not camera.isOpened():
                break
            ret, _ = camera.read()
            if ret:
                camera_name = f"摄像头 {index}"
                camera_list.append(camera_name)
            index += 1
        for camera_name in camera_list:
            self.camera_combobox.addItem(camera_name)

    def select_camera(self):
        # 释放当前摄像头资源
        self.camera.release()

        camera_index = self.camera_combobox.currentIndex()
        self.camera = cv2.VideoCapture(camera_index)

        # 设置输出画面的大小为1080x720
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # 重新获取帧的大小
        self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 停止定时器
        self.timer.stop()

        # 开始定时器以更新帧
        self.start_live_view()

    def start_live_view(self):
        if not self.timer.isActive():
            self.timer.start(30)
            self.detection_active = False  # 设置检测状态为 False
            self.start_button.setEnabled(True)  # 启用开始按钮
            self.pause_button.setEnabled(False)  # 禁用暂停按钮

    def start_detection(self):
        self.timer.start(100)
        self.detection_active = True

    def pause_detection(self):
        self.timer.stop()
        self.detection_active = False

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            if self.face_model:
                # 使用YOLOv8模型进行推理
                results = self.face_model(frame)

                # 处理检测结果
                bounding_boxes = []
                for result in results:
                    for box in result.boxes:
                        if box.cls == 0:  # 只检测人类 (类索引为0，具体取决于训练模型时的配置)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            bounding_boxes.append((x1, y1, x2, y2))

                # 更新人脸数标签
                self.face_count_label.setText(f"人脸数：{len(bounding_boxes)}")

                # 绘制检测到的人脸
                if bounding_boxes:
                    red_box_index = np.random.choice(len(bounding_boxes))
                    for i, bbox in enumerate(bounding_boxes):
                        (x1, y1, x2, y2) = bbox
                        # 增加高度和宽度 5 像素
                        x1 -= 5
                        y1 -= 5
                        x2 += 5
                        y2 += 5

                        # 调整边界，确保不超出图像范围
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1] - 1, x2)
                        y2 = min(frame.shape[0] - 1, y2)

                        # 加粗人脸框
                        thickness = 2
                        if i == red_box_index:
                            thickness = 4
                        color = (0, 0, 255) if i == red_box_index else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 调整图像大小以适应窗口
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_qimage = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0],
                                  QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(frame_qimage)

            # 根据窗口大小调整图像大小
            pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)

    def load_settings(self):
        # 加载上次使用的摄像头索引
        camera_index = self.settings.value("camera_index", defaultValue=0, type=int)
        self.camera_combobox.setCurrentIndex(camera_index)

        # 加载上次使用的模型
        model_index = self.settings.value("model_index", defaultValue=0, type=int)
        self.model_selector.setCurrentIndex(model_index)
        self.select_model(model_index)

        # 加载窗口位置
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            # 设置默认窗口位置和大小
            self.setGeometry(100, 100, 800, 480)

    def save_settings(self):
        # 保存选定的摄像头索引
        self.settings.setValue("camera_index", self.camera_combobox.currentIndex())

        # 保存选定的模型索引
        self.settings.setValue("model_index", self.model_selector.currentIndex())

        # 保存窗口位置
        self.settings.setValue("geometry", self.saveGeometry())

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    splash_pix = QPixmap('splash_image.png')
    splash = QSplashScreen(splash_pix)
    splash.show()
    app.processEvents()
    window = FaceRecognitionWidget()
    window.show()
    splash.finish(window)
    sys.exit(app.exec_())
