import os
import queue
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QSizePolicy, QMessageBox
)
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, QThread

from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # 必须放在其他 matplotlib 导入之前
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


# 加载主窗口UI文件
FORM_MAIN, _ = uic.loadUiType("./UI/main_window.ui")
# 加载图片处理窗口UI文件
FORM_IMAGE_Window, _ = uic.loadUiType("./UI/runPicture_window.ui")

FORM_Video_Window, _ = uic.loadUiType("./UI/runVideo_window.ui")

FORM_Folder_Window, _ = uic.loadUiType("./UI/runFolder_window.ui")

# ======================== 主界面 =================================#

class MainWindow(QWidget, FORM_MAIN):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # 持有“运行图片窗口”实例（防止被Python垃圾回收）
        self.image_window = None

        # 持有“运行视频窗口”实例（防止被Python垃圾回收）
        self.video_window = None

        # 持有“运行文件夹窗口”实例（防止被Python垃圾回收）
        self.folder_window = None

    def init_ui(self):
        # 进行ui控件初始化等一些操作

        # 加载Designer设计的UI到当前窗口,调用 setupUi 绑定控件到 self
        self.setupUi(self)
        # 接下来直接用ui里面的控件了

        # 绑定按钮 1 的点击事件
        self.runPictureBtn.clicked.connect(self.show_runPicture_window)

        self.runVideoBtn.clicked.connect(self.show_runVideo_window)

        self.runFolderBtn.clicked.connect(self.show_runFolder_window)

    # 自定义的槽函数，用于显示主窗口
    def show_main_window(self):
        self.show()

    def show_runPicture_window(self):
        # 首次打开时创建图片窗口（避免重复加载UI）
        if not self.image_window:
            self.image_window = ImageProcessWindow()  # 创建图片窗口实例

            # 设置为非模态窗口（允许主窗口和图片窗口同时操作）
            self.image_window.setWindowModality(0)
            # 绑定返回信号：图片窗口点击返回时触发主窗口显示
            self.image_window.back_signal.connect(self.show_main_window)

        # 隐藏主窗口（非关闭，保留状态）
        self.hide()
        # 显示图片窗口（首次创建时显示，非首次则恢复显示）
        self.image_window.show()

    def show_runVideo_window(self):
        # 首次打开时创建图片窗口（避免重复加载UI）
        if not self.video_window:
            self.video_window = VideoProcessWindow()  # 创建图片窗口实例

            # 设置为非模态窗口（允许主窗口和图片窗口同时操作）
            self.video_window.setWindowModality(0)
            # 绑定返回信号：点击返回时触发主窗口显示
            self.video_window.back_signal.connect(self.show_main_window)

        # 隐藏主窗口（非关闭，保留状态）
        self.hide()
        # 显示处理视频窗口（首次创建时显示，非首次则恢复显示）
        self.video_window.show()

    def show_runFolder_window(self):

        # 首次打开时创建图片窗口（避免重复加载UI）
        if not self.folder_window:

            self.folder_window = FolderProcessWindow()  # 创建图片窗口实例
            # 设置为非模态窗口（允许主窗口和图片窗口同时操作）
            self.folder_window.setWindowModality(0)
            # 绑定返回信号：点击返回时触发主窗口显示
            self.folder_window.back_signal.connect(self.show_main_window)

        # 隐藏主窗口（非关闭，保留状态）
        self.hide()

        # 显示处理视频窗口（首次创建时显示，非首次则恢复显示）
        self.folder_window.show()


# ======================== 主界面 =================================#

# ======================== 图片处理界面=============================#

class ImageProcessWindow(QWidget, FORM_IMAGE_Window):
    # 自定义信号：用于通知主窗口"返回"操作（跨窗口通信核心）
    back_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setupUi(self)  # 加载图片窗口的UI设计

        # 绑定返回按钮点击事件
        self.returnBtn.clicked.connect(self.go_back)

        # 绑定加载图片事件
        self.chooseBtn.clicked.connect(self.load_img)

        # 绑定推理图片事件
        self.runBtn.clicked.connect(self.load_model)

        # 定义图片路径
        self.current_image_path = ""
    # 返回主界面函数
    def go_back(self):
        # 发射返回信号（触发主窗口的show_main_window）
        self.back_signal.emit()
        # 隐藏图片窗口（非关闭，保留处理进度等状态）
        self.hide()

    # 加载原始图片到Qlabel显示
    def load_img(self):
        # 步骤 1: 创建一个 QFileDialog 对象，用于打开文件选择对话框
        file_dialog = QFileDialog()
        # 步骤 2: 调用 getOpenFileName 方法打开文件选择对话框
        # 第一个参数 self 表示该对话框的父窗口，通常为当前窗口
        # 第二个参数 "选择图片文件" 是对话框的标题
        # 第三个参数 "" 表示对话框打开时默认显示的路径，这里为空表示使用系统默认路径
        # 第四个参数 "图片文件 (*.png *.jpg *.jpeg)" 是文件过滤器，限制用户只能选择指定格式的图片文件
        # getOpenFileName 方法返回两个值，第一个是用户选择的文件路径，第二个是用户选择的文件过滤器
        file_path, _ = file_dialog.getOpenFileName(self, "选择图片文件", "", "图片文件 (*.png *.jpg *.jpeg)")

        if file_path:
            # 加载图片并显示在 label1 中
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # 以保持原始比例的方式缩放图片，使其适应 label 的大小
                scaled_pixmap = pixmap.scaled(self.img1_Lable.width(), self.img1_Lable.height(), Qt.KeepAspectRatio)
                self.img1_Lable.setPixmap(scaled_pixmap)
                # 将图片路径记录
                self.current_image_path = file_path

    def load_model(self):
        # 加载 YOLO11 模型

        # self.model = YOLO('./model/yolo11n_m.pt')
        # self.model = YOLO('./model/yolo11n_m.pt')
        self.model = YOLO('./model/yolo11m_m.pt')

        if self.current_image_path:
            # 调用 YOLOv8 进行目标检测
            self.detect_img(self.current_image_path)
        else:
            print("请先选择一张图片！")

    def detect_img(self, image_path):
        results = self.model(image_path)

        # 获取检测结果的图像
        result_image = results[0].plot()

        # 将 OpenCV 图像从 BGR 转换为 RGB
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        # 获取图像的高度、宽度和通道数
        height, width, channel = result_image.shape
        bytes_per_line = 3 * width

        # 创建 QImage 对象，复制数据
        q_image = QImage(result_image.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()

        # 将 QImage 转换为 QPixmap
        pixmap = QPixmap.fromImage(q_image)
        # 以保持原始比例的方式缩放图片，使其适应 label2 的大小
        temp = pixmap.scaled(self.img2_label.width(), self.img2_label.height(), Qt.KeepAspectRatio)
        self.img2_label.setPixmap(temp)


# ======================== 视频处理界面=============================#

class VideoProcessWindow(QWidget, FORM_Video_Window):
    # 自定义信号：用于通知主窗口"返回"操作（跨窗口通信核心）
    back_signal = pyqtSignal()

    _model_loaded = False  # 模型加载状态

    def __init__(self):
        super().__init__()
        self.setupUi(self)  # 加载图片窗口的UI

        # 初始化视频系统:定义两个线程
        self.camera_thread = None
        self.detect_thread = None

        self.init_video_system()

        # 绑定返回按钮点击事件
        self.returnBtn.clicked.connect(self.go_back)

        self.startBtn.clicked.connect(self.start_detection)

        self.stopBtn.clicked.connect(self.stop_detection)

        # 下拉框绑定信号和槽函数
        # self.comboBox.currentIndexChanged.connect(self.on_combo_box_changed)


    def go_back(self):

        # self.stop_detection()  # 停止检测再返回

        # 发射返回信号（触发主窗口的show_main_window）
        self.back_signal.emit()

        # 隐藏图片窗口（非关闭，保留处理进度等状态）
        self.hide()

    def init_video_system(self):
        """初始化视频系统（线程、模型、UI设置）"""
        # 停止之前的线程
        if self.camera_thread:
            self.camera_thread.stop()
        if self.detect_thread:
            self.detect_thread.stop()

        # 创建新的线程
        # 1. 摄像头线程（安全队列+自动重连）
        self.camera_thread = CameraThread()

        self.camera_thread.frame_signal.connect(self._update_frame)

        self.camera_thread.status_signal.connect(self._update_status)


        # 2. 推理线程（YOLOv11+缺陷标注）
        self.detect_thread = DetectionThread("./model/yolo11s_m.pt")
        self.detect_thread.result_signal.connect(self._display_result)

        # 3. UI控件配置
        # self.video_label.setScaledContents(False)  # 禁用自动缩放
        # self.video_label.setSizePolicy(
        #     QSizePolicy.Expanding,  # 随窗口拉伸
        #     QSizePolicy.Expanding
        # )
        self.video_label.setStyleSheet("background: #000; border-radius: 8px;")

    def start_detection(self):
        """启动检测流程（摄像头+推理）"""
        print("开始启动推理")
        if not self._model_loaded:
            if not self.detect_thread.load_model():
                self._update_status("❌ 模型加载失败，请检查路径！")
                return

        if self.camera_thread.isRunning():
            return

        self._update_status("启动中...")

        self.camera_thread.start()

        self.detect_thread.set_camera_thread(self.camera_thread)

        # start()自动触发 DetectionThread 子类的 run() 方法在新线程中执行
        self.detect_thread.start()

        self.startBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        print("推理成功启动")

    def stop_detection(self):
        """停止检测流程（释放资源）"""
        if self.camera_thread:  # 空指针保护
            self.camera_thread.stop()
        if self.detect_thread:
            self.detect_thread.stop()

        self.video_label.clear()

        self._update_status("已停止")

        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)

    def _update_frame(self, frame):
        """原始帧预处理（发送给推理线程）"""
        if self.detect_thread.isRunning():
            self.detect_thread.input_queue.put(frame)

    def _display_result(self, result_image):

        """显示带标注的视频帧（主线程更新）"""
        if result_image is None:
            print("什么都没有啊")
            return

        print("得到处理结果帧")


        # 获取图像的高度、宽度和通道数
        height, width, channel = result_image.shape
        bytes_per_line = 3 * width

        # 创建 QImage 对象，复制数据
        q_image = QImage(result_image.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()

        # 将 QImage 转换为 QPixmap
        pixmap = QPixmap.fromImage(q_image)
        # 以保持原始比例的方式缩放图片，使其适应 label 的大小
        temp = pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)

        self.video_label.setPixmap(temp)

    def _update_status(self, text):
        """更新状态栏（带颜色标记）"""
        color = "#4CAF50" if "✅" in text else "#FF5722" if "❌" in text else "#666"
        self.status_label.setText(f'<span style="color:{color}">{text}</span>')



# ========== 底层线程实现 ==========
class CameraThread(QThread):
    # 发送图像帧
    frame_signal = pyqtSignal(np.ndarray)
    # 发送状态文本
    status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = False
        self.retries = 3  # 自动重连次数

    def run(self):
        self.running = True
        for _ in range(self.retries):
            self.cap = cv2.VideoCapture(0)  # 0为默认摄像头
            if self.cap.isOpened():
                self.status_signal.emit("✅ 摄像头已连接")
                break
            self.status_signal.emit(f"❌ 连接失败（重试{_+1}/{self.retries}）")
            self.sleep(1)
        else:
            return

        while self.running:
            """ret: 布尔值（True= 成功，False= 帧丢失 / 设备断开）, frame: np.ndarray（BGR 格式，工业缺陷检测的原始数据）"""
            ret, frame = self.cap.read()    # 核心帧采集接口
            if not ret:
                self.status_signal.emit("⚠️ 丢失帧，正在恢复...")
                continue
            # emit() 方法里的参数会直接传递给和信号绑定的槽函数
            self.frame_signal.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 转换为RGB,并且将frame发送到主线程了
            self.msleep(33)  # 约30fps （1000ms/33ms≈30.3帧）


    def stop(self):
        self.running = False
        # self.wait(1000) 中的 self 是摄像头线程实例，但 wait 方法的作用是让 调用它的线程（即主线程） 等待，
        # 直到摄像头线程的 run 方法执行完毕，或超时 1000 毫秒。
        self.wait(1000)
        if self.cap:
            self.cap.release()

class DetectionThread(QThread):
    result_signal = pyqtSignal(np.ndarray)

    #队列未满，put()立即将数据放入队列，不阻塞；队列已满，put()阻塞线程，直到队列中有空间（即其他线程调用 get() 取出数据）。
    input_queue = queue.Queue(maxsize=3)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.model = None
        self.camera_thread = None
        self.running = False

    def load_model(self):
        """加载YOLO模型（带异常处理）"""
        try:
            self.model = YOLO(self.model_path)
            self.model.fuse()  # 模型加速
            self._model_loaded = True
            print("模型加载成功")
            return True
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return False

    def set_camera_thread(self, camera_thread):
        self.camera_thread = camera_thread

    def run(self):
        self.running = True
        while self.running and self.camera_thread:
            if self.input_queue.empty():
                self.msleep(10)
                continue

            frame = self.input_queue.get()
            if frame is None:
                continue

            print("得到一个帧，并且执行推理")
            # 推理核心（工业优化）

            results = self.model(frame)
            print("处理完成")

            # 绘制检测结果到图像上，不需要手动设置检测框
            result_image = results[0].plot()


            # 将 OpenCV 图像从 BGR 转换为 RGB
            # result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

            self.result_signal.emit(result_image)

            # Windows 限制：PyTorch 的 GPU 张量无法跨线程传递（内存地址空间隔离）
            # self.result_signal.emit(results)

    def stop(self):
        self.running = False  # 设置停止标志
        self.wait(1000)  # 等待推理线程结束，无参数时，阻塞直到线程结束
        # 可选：清空队列或释放模型资源
        # 清空队列
        with self.input_queue.mutex:  # 线程安全地清空队列
            self.input_queue.queue.clear()

        # 释放模型（GPU 场景）
        if self.model:
            del self.model  # 释放模型占用的内存（如 GPU 显存）
            torch.cuda.empty_cache()  # 清理 GPU 显存

# ======================== 文件夹批量处理图片界面=============================#
class FolderProcessWindow(QWidget, FORM_Folder_Window):
    # 自定义信号：用于通知主窗口"返回"操作（跨窗口通信核心）
    back_signal = pyqtSignal()
    # 主线程（UI线程）更新进度条
    update_progress_signal = pyqtSignal()
    # 更新处理文件
    update_text_signal = pyqtSignal(str)
    # 开始绘制图表信号
    generate_and_save_charts_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        # self是主窗口类的实例

        self.setupUi(self)  # 加载图片窗口的UI设计
        self.folder_path = None  # 选择的文件夹路径
        # 存储所有创建的推理线程对象，每当创建一个新的推理线程时，线程对象会被添加到这个列表中
        self.inference_threads = []

        # 线程间通信的同步工具，用于向所有推理线程广播停止信号。
        self.stop_event = threading.Event()
        self.model_path = None
        self.confidence = 0.5
        # 触发CUDA初始化：预防多线程同时进行cuda初始化，触发多次上下文创建，导致死锁。
        if torch.cuda.is_available():
            _ = torch.tensor([1.0], device='cuda')
            torch.cuda.empty_cache()
            print(f"线程 {threading.get_ident()} 已清理显存")

        # 添加推理专用锁
        self.inference_lock = threading.Lock()

        # 添加类别统计字典和锁
        # 统计一组图片出现的异常类别和个数
        self.class_counts = defaultdict(int)
        self.class_counts_lock = threading.Lock()

        self.chooseBtn.clicked.connect(self.select_folder)
        self.startBtn.clicked.connect(self.start_inference)
        self.stopBtn.clicked.connect(self.stop_inference)

        # 绑定返回按钮点击事件
        self.returnBtn.clicked.connect(self.go_back)

        # 信号连接下拉框和滑轨
        self.comboBox.currentIndexChanged.connect(self.select_model)
        # 下拉框有个小bug，检测到选中内容发生改变之后，才会执行select_model()，所以初始化时先手动执行一下select_model()，不然没有选中模型
        self.select_model()

        self.confSlider.valueChanged.connect(self.set_confidence)

        # 初始化置信度滑轨
        self.init_conSlider()
        # 连接信号和槽：更新进度条，追加文本
        self.update_progress_signal.connect(self.update_progress)
        self.update_text_signal.connect(self.append_text)
        self.generate_and_save_charts_signal.connect(self.generate_and_save_charts)

        self.progressBar.setValue(0)

    # 返回主界面函数
    def go_back(self):
        # 发射返回信号（触发主窗口的show_main_window）
        self.back_signal.emit()
        # 隐藏图片窗口（非关闭，保留处理进度等状态）
        self.hide()

    # 得到模型的路径，路径给每个线程，每个线程自己负责加载模型
    def select_model(self):
        model_name = self.comboBox.currentText()
        self.model_path = os.path.join('model', model_name)
        if not os.path.exists(self.model_path):
            QtWidgets.QMessageBox.critical(self, "错误", f"本地未找到模型文件: {self.model_path}")
            return
        print(f"当前选择模型路径: {self.model_path}")

    # 设置模型的置信率
    def set_confidence(self):
        print("当前选择的置信率为：")
        print(self.confSlider.value() / 100)
        self.confidence = self.confSlider.value() / 100
        self.update_confLabel()

    def init_conSlider(self):
        # 设置 QSlider 的取值范围，这里设置为 0 到 100
        self.confSlider.setRange(1, 100)

    def update_confLabel(self):
        # 将滑块的整数值映射到 0.01 到 1.00 的范围
        value = (self.confSlider.value()) / 100
        self.conf_label.setText(f"当前取值: {value:.2f}")

    def update_progress(self):
        current = self.progressBar.value()
        self.progressBar.setValue(current + 1)

    def append_text(self, text):
        self.plainTextEdit.appendPlainText(text)


    def select_folder(self):
        # 选择文件夹
        self.folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, '选择文件夹')
        if self.folder_path:
            print(f"选择的文件夹: {self.folder_path}")
            # 文件夹路径显示在前端label
            self.folder_label.setText(self.folder_path)

    def start_inference(self):
        if not self.model_path:
            QtWidgets.QMessageBox.critical(self, "错误", "请先选择模型")
            return
        if not self.folder_path:
            QtWidgets.QMessageBox.critical(self, "错误", "请先选择图片文件夹")
            return
        # 允许线程执行推理
        self.stop_event.clear()

        # 获取所有图片文件
        image_files = []
        # 递归遍历选择的文件夹及其子文件夹：root当前正在遍历的目录的完整路径、dirs当前目录下的所有子目录的名称列表、files当前目录下的所有文件的名称列表。
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))

        print("成功读取所有图片")

        # 创建结果文件夹
        base_folder_name = 'detect_result'
        index = 1
        result_folder = os.path.join(os.path.dirname(self.folder_path), base_folder_name)
        while os.path.exists(result_folder):
            result_folder = os.path.join(os.path.dirname(self.folder_path), f'{base_folder_name}_{index}')
            index += 1
        os.makedirs(result_folder, exist_ok=True)
        print("成功创建结果文件夹")

        # 初始化进度条：在这里初始化，start_inference只会执行一次，它里面的progess_image会执行多次
        num_images = len(image_files)
        self.progressBar.setRange(0, num_images)
        self.progressBar.setValue(0)
        print("进度条初始化完成")

        # 创建线程池时预加载模型参数
        model_path = self.model_path  # 获取主线程中的模型路径
        confidence = self.confidence  # 获取置信度参数
        # 创建带模型缓存的线程池
        # 线程池里最多可以同时运行 4 个工作线程，并且在每个工作线程启动时，会调用 self.worker_init 函数进行初始化操作，
        # 同时会将 model_path 作为参数传递给这个初始化函数。
        self.thread_local = threading.local()
        self.executor = ThreadPoolExecutor(max_workers=4, initializer=self.worker_init, initargs=(model_path,))
        self.futures = []
        print(f"创建的线程池大小为: {self.executor._max_workers}")
        # print(f"系统拥有的CPU核心数: {os.cpu_count()}")

        # 在创建线程池后添加：
        self.class_counts.clear()  # 清空历史统计

        # 函数内部再定义一个函数：这个函数一次只处理一张图片
        def process_image(image_path):
            if self.stop_event.is_set():
                return
            try:
                # 从线程本地存储获取模型
                if not hasattr(self.thread_local, "model"):
                    print("线程模型未初始化")
                    raise RuntimeError("线程模型未初始化")

                # 执行推理
                print("开始执行推理...")
                # results = self.model(image_path, conf=self.confidence)
                # self.thread_local 是 threading.local() 对象，它为每个线程提供独立的存储空间。
                with self.inference_lock:
                    results = self.thread_local.model(image_path, conf=self.confidence)
                print("一张图片完成推理...")
                # 取出文件路径里面的文件名
                image_name = os.path.basename(image_path)
                # 构建推理结果文本信息
                result_text = f"{image_name}\n"
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.thread_local.model.names[class_id]
                        result_text += f"检测到的类别: {class_name}, 置信度: {confidence:.2f}\n"

                        # 使用锁安全更新统计
                        with self.class_counts_lock:
                            self.class_counts[class_name] += 1
                        # 控制台打印信息
                        print(result_text)
                # 更新文本显示
                self.update_text_signal.emit(result_text)

                # 保存结果图片
                result_img = results[0].plot()
                print("得到一个推理图片...")
                from PIL import Image
                # 保存推理结果
                print("保存一个推理图片...")
                Image.fromarray(result_img[..., ::-1]).save(os.path.join(result_folder, image_name))
                print("保存成功...")
                self.update_progress_signal.emit()
            except Exception as e:
                print(f"处理图片 {image_path} 时出错: {e}")

            finally:
                # 确保进度条更新
                self.update_progress_signal.emit()

        # 提交所有任务到线程池
        for image_path in image_files:
            print("提交任务到线程池...")
            print(f"选择的图片路径：{image_path}")
            # self.executor线程池 的submit()方法将一个可调用对象（process_image 函数）和其参数（image_path）提交到线程池中进行执行。
            # submit() 方法会立即返回一个 Future 对象，这个对象代表了异步执行的任务。
            future = self.executor.submit(process_image, image_path)
            self.futures.append(future)

        # 等待所有任务完成
        # shutdown 方法用于关闭线程池或者进程池。wait=True 表示在关闭之前会等待所有已提交的任务执行完毕。
        self.executor.shutdown(wait=True)

        # 生成并保存图表
        self.generate_and_save_charts_signal.emit(result_folder)

    # 线程初始化函数
    def worker_init(self, model_path):
        # 每个线程初始化时创建独立模型
        # 将线程本地存储挂载到类实例属性上，多个线程同时操作 self.thread_local 可能产生意外覆盖
        # self.thread_local.model = YOLO(model_path)
        # 所有线程都通过self.thread_local属性访问

        try:
            # 添加显存清理（如果使用GPU）  多线程使用这个让我调bug调了三小时
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            #     print(f"线程 {threading.get_ident()} 已清理显存")

            print(f"线程 {threading.get_ident()} 正在加载模型...")
            self.thread_local.model = YOLO(model_path)
            if torch.cuda.is_available():
                self.thread_local.model.to('cuda')
            # 使用cpu验证是否会出现多线程卡住状态
            # self.thread_local.model = YOLO(model_path).to('cpu')  # 添加设备指定
            # self.thread_local.model.fuse()  # 如果需要加速
            print(f"线程 {threading.get_ident()} 模型加载完成")
            print(f"线程本地模型内存地址: {id(self.thread_local.model)}")
            print(f"模型类别字典: {self.thread_local.model.names}")  # 验证是否能获取names
            # 检查设备状态
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"线程 {threading.get_ident()} 使用设备: {device}")
        except Exception as e:
            print(f"线程初始化失败: {str(e)}")
            raise

    def stop_inference(self):
        """停止推理时触发的槽函数"""
        self.stop_event.set()
        if self.executor:
            # 立即关闭线程池并取消所有排队任务
            self.executor.shutdown(wait=False, cancel_futures=True)
            print("推理已终止")

    # 绘制这一批图片的异常个数信息（智能分析）
    def generate_and_save_charts(self, result_folder):
        # 确保在主线程执行（如果涉及GUI操作）

        if not self.class_counts:
            print("没有检测到任何类别，跳过图表生成")
            return
        print("开始绘制图表")
        # 准备数据
        labels = list(self.class_counts.keys())
        counts = list(self.class_counts.values())

        # 这两行解决plt.title设置中文时不能正常显示
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # ==================== 饼状图单独保存 ====================
        try:
            plt.figure(figsize=(8, 8))  # 创建独立画布
            plt.pie(counts,
                    labels=labels,
                    autopct=lambda p: f'{p:.1f}%\n({int(p * sum(counts) / 100)})',  # 显示百分比和实际数量
                    startangle=140,
                    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                    textprops={'fontsize': 10})
            plt.title('异常类别分布饼状图', fontsize=14, pad=20)

            # 保存饼图
            pie_path = os.path.join(result_folder, "category_pie.png")
            plt.savefig(pie_path, dpi=300, bbox_inches='tight')
            plt.close()  # 显式关闭当前图表

            print(f"饼状图已保存至: {pie_path}")

            # 显示图片到label控件中
            pixmap = QPixmap(pie_path)
            if not pixmap.isNull():
                # 保持宽高比缩放适应标签
                scaled_pixmap = pixmap.scaled(
                    self.img1_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.img1_label.setPixmap(scaled_pixmap)
                self.img1_label.setAlignment(Qt.AlignCenter)
                print("饼状图已显示在界面")
            else:
                print("错误：饼状图加载失败")


        except Exception as e:
            print(f"保存饼状图失败: {str(e)}")

            # ==================== 柱状图单独保存 ====================
        try:
            plt.figure(figsize=(12, 6))  # 创建新画布
            bars = plt.bar(labels, counts, color='#2c7fb8', edgecolor='black')
            plt.xlabel('异常类别', fontsize=12)
            plt.ylabel('出现次数', fontsize=12)
            plt.title('异常类别分布柱状图', fontsize=14, pad=20)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)

            # 添加数值标签
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2,
                         yval + max(counts) * 0.02,  # 避免遮挡柱顶
                         f'{yval}',
                         ha='center',
                         va='bottom',
                         fontsize=9)

            # 调整边距
            plt.subplots_adjust(bottom=0.25)  # 为长标签留出空间

            # 保存柱状图
            bar_path = os.path.join(result_folder, "category_bar.png")
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"柱状图已保存至: {bar_path}")

            # 显示图片在label中
            pixmap = QPixmap(bar_path)
            if not pixmap.isNull():
                # 保持宽高比缩放适应标签
                scaled_pixmap = pixmap.scaled(
                    self.img2_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.img2_label.setPixmap(scaled_pixmap)
                self.img2_label.setAlignment(Qt.AlignCenter)
                print("柱状图已显示在界面")
            else:
                print("错误：柱状图加载失败")
        except Exception as e:
            print(f"保存柱状图失败: {str(e)}")
        print(f"图表生成完成，文件位于: {result_folder}")


if __name__ == '__main__':
    # 创建一个对象，实例化，传参
    app = QApplication(sys.argv)

    w = MainWindow()

    # 展示窗口
    w.show()

    # 进入程序的主循环，并通过exit函数确保主循环安全结束(该释放资源的一定要释放)
    sys.exit(app.exec_())
