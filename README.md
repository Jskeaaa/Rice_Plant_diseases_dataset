# 水稻叶片疾病分类

使用深度学习（ResNet-50模型）对水稻叶片疾病进行分类的项目。华宇 LGJ (coding)

## 项目概述

该项目使用ResNet-50卷积神经网络模型对水稻叶片疾病进行分类，包括以下三种疾病类别：
- 细菌性叶枯病（Bacterial Blight）
- 褐斑病（Brown Spot）
- 叶疤病（Leaf Smut）

## 数据集

数据集包含三个疾病类别的水稻叶片图像，位于`rice leaf diseases dataset`文件夹中。

## 项目结构

```
Rice_Plant_diseases_dataset/
│
├── rice leaf diseases dataset/  # 原始数据集
│   ├── Bacterialblight/
│   ├── Brownspot/
│   └── Leafsmut/
│
├── src/
│   ├── data_preprocessing.py    # 数据预处理和加载模块
│   ├── model.py                 # 模型定义模块
│   ├── train.py                 # 训练脚本
│   ├── evaluate.py              # 评估脚本
│   └── utils.py                 # 工具函数
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # 数据探索分析
│
├── requirements.txt             # 项目依赖
└── README.md                    # 项目说明
```

## 安装与使用

1. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

2. 运行训练：
   ```
   python src/train.py
   ```

3. 评估模型：
   ```
   python src/evaluate.py
   ```

## 模型架构

本项目使用了基于ResNet-50架构修改得到的ResNet-59模型，通过添加额外的残差块实现，最终用于3类水稻叶片疾病的分类任务。
