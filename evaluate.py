import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_preprocessing import create_dataloaders, get_transforms
from model import load_model
from utils import (
    create_dirs, plot_confusion_matrix, print_classification_report,
    visualize_predictions, visualize_model_attention
)

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计损失和准确率
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集预测结果和标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
    
    test_loss = test_loss / total
    test_acc = 100. * correct / total
    
    return test_loss, test_acc, all_preds, all_labels

def compute_metrics(y_true, y_pred, class_names):
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    
    # 计算每个类别的精确率、召回率和F1分数
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    # 计算宏平均和微平均
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    
    # 构建指标字典
    metrics_dict = {
        'accuracy': accuracy,
        'class_metrics': {},
        'macro_avg': {
            'precision': precision_macro,
            'recall': recall_macro,
            'f1': f1_macro
        },
        'micro_avg': {
            'precision': precision_micro,
            'recall': recall_micro,
            'f1': f1_micro
        }
    }
    
    # 添加每个类别的指标
    for i, class_name in enumerate(class_names):
        metrics_dict['class_metrics'][class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i]
        }
    
    return metrics_dict

def main():
    parser = argparse.ArgumentParser(description="评估水稻叶片疾病分类模型")
    parser.add_argument('--data_dir', type=str, default="../rice leaf diseases dataset",
                        help="数据集路径")
    parser.add_argument('--model_path', type=str, default="results/best_model.pth",
                        help="模型权重文件路径")
    parser.add_argument('--result_dir', type=str, default="evaluation_results",
                        help="评估结果保存路径")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="批处理大小")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="数据加载的工作线程数")
    parser.add_argument('--sample_images', type=int, default=6,
                        help="可视化的样本数量")
    parser.add_argument('--visualize_attention', action='store_true',
                        help="是否可视化模型注意力")
    
    args = parser.parse_args()
    
    # 创建结果保存目录
    create_dirs([args.result_dir])
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # 类别名称
    class_names = list(class_to_idx.keys())
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # 加载模型
    model = load_model(args.model_path, num_classes=len(class_names), device=device)
    print(f"模型已加载: {args.model_path}")
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 在测试集上评估模型
    test_loss, test_acc, all_preds, all_labels = evaluate_model(
        model, test_loader, criterion, device
    )
    print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%")
    
    # 计算各种评估指标
    metrics = compute_metrics(all_labels, all_preds, class_names)
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"宏平均精确率: {metrics['macro_avg']['precision']:.4f}")
    print(f"宏平均召回率: {metrics['macro_avg']['recall']:.4f}")
    print(f"宏平均F1分数: {metrics['macro_avg']['f1']:.4f}")
    
    # 绘制混淆矩阵
    cm_path = os.path.join(args.result_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        all_labels, all_preds, class_names, 
        normalize=False, save_path=cm_path
    )
    
    # 归一化混淆矩阵
    norm_cm_path = os.path.join(args.result_dir, "normalized_confusion_matrix.png")
    plot_confusion_matrix(
        all_labels, all_preds, class_names, 
        normalize=True, save_path=norm_cm_path
    )
    
    # 打印分类报告
    report_path = os.path.join(args.result_dir, "classification_report.csv")
    print_classification_report(all_labels, all_preds, class_names, save_path=report_path)
    
    # 可视化预测结果
    visualize_predictions(
        model, test_loader, class_names, device, 
        num_images=args.sample_images, 
        save_dir=args.result_dir
    )
    
    # 可视化模型注意力（Grad-CAM）
    if args.visualize_attention:
        print("可视化模型注意力...")
        
        # 获取每个类别的一个样本
        test_transform = get_transforms('test')
        attention_dir = os.path.join(args.result_dir, "attention_maps")
        create_dirs([attention_dir])
        
        # 设置Grad-CAM可视化
        for class_idx, class_name in enumerate(class_names):
            # 尝试找到该类别的样本
            found_sample = False
            for inputs, labels in test_loader:
                batch_indices = (labels == class_idx).nonzero(as_tuple=True)[0]
                if len(batch_indices) > 0:
                    sample_idx = batch_indices[0].item()
                    sample_img = inputs[sample_idx]
                    
                    # 将张量转换为PIL图像
                    sample_img_np = sample_img.numpy().transpose((1, 2, 0))
                    # 反标准化
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    sample_img_np = std * sample_img_np + mean
                    sample_img_np = np.clip(sample_img_np * 255, 0, 255).astype(np.uint8)
                    
                    # 保存样本图像
                    sample_path = os.path.join(attention_dir, f"{class_name}_sample.jpg")
                    import cv2
                    cv2.imwrite(sample_path, cv2.cvtColor(sample_img_np, cv2.COLOR_RGB2BGR))
                    
                    # 生成Grad-CAM热力图
                    try:
                        visualize_model_attention(
                            model, sample_path, class_names, test_transform, 
                            target_layer_name='layer4', device=device
                        )
                        found_sample = True
                        break
                    except Exception as e:
                        print(f"为类别 {class_name} 生成Grad-CAM时出错: {e}")
                        continue
                
                if found_sample:
                    break
            
            if not found_sample:
                print(f"未找到类别 {class_name} 的样本进行可视化")

if __name__ == "__main__":
    main() 