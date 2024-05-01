import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer


class FaceRecognitionWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CV-RollCall")  # 设置窗口标题
        self.resize(800, 480)  # 设置窗口大小

        # 加载预训练的人脸检测模型
        self.face_model = cv2.dnn.readNetFromCaffe("models/deploy.prototxt",
                                                   "models/res10_300x300_ssd_iter_140000_fp16.caffemodel")

        # 摄像头选择
        self.camera_combobox = QComboBox()  # 创建摄像头选择下拉框
        self.populate_cameras()  # 填充摄像头列表
        self.camera_combobox.currentIndexChanged.connect(self.select_camera)  # 连接选择摄像头的信号与槽函数

        # 摄像头
        self.camera = cv2.VideoCapture(0)  # 打开摄像头
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)  # 设置摄像头帧宽度
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 设置摄像头帧高度
        self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取帧宽度
        self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取帧高度

        # 定时器
        self.timer = QTimer()  # 创建定时器
        self.timer.timeout.connect(self.update_frame)  # 连接定时器超时信号与更新帧函数

        # 添加按钮
        self.start_button = QPushButton("开始")  # 创建开始按钮
        self.start_button.clicked.connect(self.start_detection)  # 连接开始按钮的点击信号与开始检测函数

        self.pause_button = QPushButton("停止")  # 创建暂停按钮
        self.pause_button.clicked.connect(self.pause_detection)  # 连接暂停按钮的点击信号与暂停检测函数
        self.pause_button.setEnabled(False)  # 设置暂停按钮不可用

        # 图片显示区域
        self.image_label = QLabel()  # 创建显示图片的标签
        self.image_label.setFixedSize(self.frame_width, self.frame_height)  # 设置标签大小

        # 设置布局
        layout = QVBoxLayout()  # 创建垂直布局
        layout.addWidget(self.image_label)  # 将图片标签添加到布局中
        layout.addWidget(self.camera_combobox)  # 将摄像头选择下拉框添加到布局中
        layout.addWidget(self.start_button)  # 将开始按钮添加到布局中
        layout.addWidget(self.pause_button)  # 将暂停按钮添加到布局中

        self.setLayout(layout)  # 设置窗口布局

    def populate_cameras(self):
        camera_list = []  # 存储摄像头设备名称列表
        index = 0
        while True:
            camera = cv2.VideoCapture(index)
            if not camera.isOpened():
                break
            else:
                ret, _ = camera.read()
                if ret:
                    camera_name = f"摄像头 {index}"  # 生成摄像头名称
                    camera_list.append(camera_name)  # 将摄像头名称添加到列表中
            index += 1
        for camera_name in camera_list:
            self.camera_combobox.addItem(camera_name)  # 将摄像头名称添加到下拉框中

    def select_camera(self):
        camera_index = self.camera_combobox.currentIndex()  # 获取选择的摄像头索引
        self.camera = cv2.VideoCapture(camera_index)  # 打开选择的摄像头

        # 设置摄像头帧宽度和高度
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.timer.stop()  # 暂停定时器
        self.start_button.setEnabled(True)  # 启用开始按钮
        self.pause_button.setEnabled(False)  # 禁用暂停按钮

    def start_detection(self):
        self.timer.start(100)  # 开始定时器
        self.start_button.setEnabled(False)  # 禁用开始按钮
        self.pause_button.setEnabled(True)  # 启用暂停按钮

    def pause_detection(self):
        self.timer.stop()  # 暂停定时器
        self.start_button.setEnabled(True)  # 启用开始按钮
        self.pause_button.setEnabled(False)  # 禁用暂停按钮

    def update_frame(self):
        ret, frame = self.camera.read()  # 读取摄像头帧
        if ret:
            h, w = frame.shape[:2]  # 获取帧高度和宽度

            # 准备用于人脸检测的图像
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            # 通过网络进行人脸检测
            self.face_model.setInput(blob)
            detections = self.face_model.forward()

            # 初始化标记
            red_box_index = None
            green_box_indices = []

            # 遍历检测到的人脸
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # 过滤掉置信度较低的检测结果
                if confidence > 0.3:
                    # 计算边界框的坐标
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # 标记所有检测到的人脸使用绿框
                    green_box_indices.append(i)

            # 如果有检测到人脸，随机选择一个使用红框标记
            if green_box_indices:
                red_box_index = np.random.choice(green_box_indices)

            # 再次遍历检测到的人脸，绘制边界框
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # 过滤掉置信度较低的检测结果
                if confidence > 0.3:
                    # 计算边界框的坐标
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # 根据是否是红框标记的人脸确定颜色
                    if i == red_box_index:
                        color = (0, 0, 255)  # 红色框标记
                    else:
                        color = (0, 255, 0)  # 绿色框标记

                    # 在人脸周围绘制边界框
                    border_thickness = 3  # 边框粗细
                    box_increase = 8  # 边框增加的像素值

                    cv2.rectangle(frame, (startX - box_increase, startY - box_increase),
                                  (endX + box_increase, endY + box_increase), color, border_thickness)

            # 将帧转换为 QImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bytes_per_line = frame_rgb.shape[1] * 3
            convert_to_qt_format = QImage(frame_rgb.data, self.frame_width, self.frame_height, bytes_per_line,
                                          QImage.Format_RGB888)
            p = QPixmap.fromImage(convert_to_qt_format)
            self.image_label.setPixmap(p)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionWidget()
    window.show()
    sys.exit(app.exec_())
