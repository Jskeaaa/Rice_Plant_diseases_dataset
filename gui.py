import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget,
                           QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
                           QFileDialog, QProgressBar, QMessageBox, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import torch
from PIL import Image
import numpy as np

from model import load_model
from data_preprocessing import get_transforms
from utils import set_seed

class PredictionThread(QThread):
    """预测线程类，用于处理耗时的预测操作"""
    finished = pyqtSignal(str)  # 预测完成信号
    progress = pyqtSignal(int)  # 进度信号

    def __init__(self, model, transform, image_path, is_batch=False):
        super().__init__()
        self.model = model
        self.transform = transform
        self.image_path = image_path
        self.is_batch = is_batch
        # 保存模型所在的设备
        self.model.device = next(model.parameters()).device

    def run(self):
        try:
            if not self.is_batch:
                # 单张图片处理
                result = self.predict_single_image(self.image_path)
                self.finished.emit(result)
            else:
                # 批量处理
                results = self.predict_batch_images(self.image_path)
                self.finished.emit(results)
        except Exception as e:
            self.finished.emit(f"错误：{str(e)}")

    def predict_single_image(self, image_path):
        """预测单张图片"""
        try:
            # 加载图片
            image = Image.open(image_path).convert('RGB')
            # 应用转换
            image_tensor = self.transform(image).unsqueeze(0)
            
            # 将输入数据移动到与模型相同的设备上
            image_tensor = image_tensor.to(self.model.device)
            
            # 进行预测
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                
                # 获取预测概率
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                # 获取预测结果
                pred_class = predicted.item()
                pred_prob = probabilities[pred_class].item()
                
                # 返回结果
                class_names = ['细菌性叶枯病', '褐斑病', '叶疤病']
                result = f"预测结果：{class_names[pred_class]}\n置信度：{pred_prob:.2%}"
                return result
        except Exception as e:
            return f"预测出错：{str(e)}"

    def predict_batch_images(self, folder_path):
        """批量预测文件夹中的图片"""
        try:
            results = []
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_files = len(image_files)
            
            for i, image_file in enumerate(image_files):
                image_path = os.path.join(folder_path, image_file)
                result = self.predict_single_image(image_path)
                results.append(f"文件：{image_file}\n{result}\n")
                
                # 更新进度
                progress = int((i + 1) / total_files * 100)
                self.progress.emit(progress)
            
            return "\n".join(results)
        except Exception as e:
            return f"批量处理出错：{str(e)}"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水稻叶片疾病分类系统")
        self.setGeometry(100, 100, 800, 600)

        # 加载模型
        self.load_model()

        # 创建主界面
        self.init_ui()

    def load_model(self):
        """加载模型"""
        try:
            set_seed()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = load_model('results/best_model.pth', num_classes=3, device=self.device)
            self.transform = get_transforms('test')
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败：{str(e)}")
            sys.exit(1)

    def init_ui(self):
        """初始化用户界面"""
        # 创建中心部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 创建选项卡
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # 添加单张图片处理选项卡
        single_tab = QWidget()
        tabs.addTab(single_tab, "单张图片处理")
        self.init_single_tab(single_tab)

        # 添加批量处理选项卡
        batch_tab = QWidget()
        tabs.addTab(batch_tab, "批量处理")
        self.init_batch_tab(batch_tab)

    def init_single_tab(self, tab):
        """初始化单张图片处理选项卡"""
        layout = QVBoxLayout(tab)

        # 创建图片显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("border: 2px solid #cccccc;")
        
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # 创建按钮和结果显示区域
        button_layout = QHBoxLayout()
        self.select_button = QPushButton("选择图片")
        self.select_button.clicked.connect(self.select_single_image)
        self.process_button = QPushButton("处理图片")
        self.process_button.clicked.connect(self.process_single_image)
        self.process_button.setEnabled(False)
        
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.process_button)
        layout.addLayout(button_layout)

        # 结果显示
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 14px; margin: 10px;")
        layout.addWidget(self.result_label)

    def init_batch_tab(self, tab):
        """初始化批量处理选项卡"""
        layout = QVBoxLayout(tab)

        # 创建按钮
        self.batch_select_button = QPushButton("选择文件夹")
        self.batch_select_button.clicked.connect(self.select_batch_folder)
        layout.addWidget(self.batch_select_button)

        # 进度条
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # 结果显示区域
        self.batch_result_label = QLabel()
        self.batch_result_label.setAlignment(Qt.AlignLeft)
        self.batch_result_label.setWordWrap(True)
        
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidget(self.batch_result_label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

    def select_single_image(self):
        """选择单张图片"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg)")
        
        if file_name:
            self.current_image_path = file_name
            self.display_image(file_name)
            self.process_button.setEnabled(True)

    def display_image(self, image_path):
        """显示图片"""
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def process_single_image(self):
        """处理单张图片"""
        self.process_button.setEnabled(False)
        self.result_label.setText("正在处理...")
        
        # 创建预测线程
        self.pred_thread = PredictionThread(
            self.model, self.transform, self.current_image_path)
        self.pred_thread.finished.connect(self.on_single_prediction_complete)
        self.pred_thread.start()

    def on_single_prediction_complete(self, result):
        """单张图片预测完成回调"""
        self.result_label.setText(result)
        self.process_button.setEnabled(True)

    def select_batch_folder(self):
        """选择批处理文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            self.batch_select_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self.batch_result_label.setText("正在处理...")
            
            # 创建预测线程
            self.batch_thread = PredictionThread(
                self.model, self.transform, folder_path, is_batch=True)
            self.batch_thread.finished.connect(self.on_batch_prediction_complete)
            self.batch_thread.progress.connect(self.update_progress)
            self.batch_thread.start()

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def on_batch_prediction_complete(self, results):
        """批量预测完成回调"""
        self.batch_result_label.setText(results)
        self.batch_select_button.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 