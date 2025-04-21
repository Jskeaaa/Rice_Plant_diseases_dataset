import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

# 设置CUDA相关配置
if torch.cuda.is_available():
    # 设置CUDA版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
    # 优化CUDA性能
    torch.backends.cudnn.benchmark = True  # 启用cudnn基准测试
    torch.backends.cudnn.deterministic = True  # 确保可重复性
    torch.backends.cudnn.enabled = True  # 启用cudnn
    torch.cuda.set_device(0)  # 设置默认GPU设备
    
    # 设置内存分配器
    torch.cuda.empty_cache()  # 清空缓存
    torch.cuda.memory.empty_cache()  # 清空内存缓存
else:
    print("警告：未检测到可用的GPU，将使用CPU进行训练")

from data_preprocessing import create_dataloaders, set_seed
from model import get_model
from utils import (
    create_dirs, save_model_and_history, plot_training_history,
    early_stopping, adjust_learning_rate
)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计损失和准确率
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """在验证集上评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计损失和准确率
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def train_model(data_dir, save_dir='results', num_epochs=30, batch_size=32, 
               learning_rate=0.001, weight_decay=1e-4, num_workers=4,
               patience=5, use_early_stopping=True, use_scheduler=True):
    """训练模型的主函数"""
    # 创建保存目录
    create_dirs([save_dir])
    
    # 设置随机种子
    set_seed()
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU内存使用情况: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"GPU内存缓存: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
    
    # 加载数据
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        data_dir, batch_size=batch_size, num_workers=num_workers
    )
    print(f"类别映射: {class_to_idx}")
    
    # 类别名称（用于可视化）
    class_names = list(class_to_idx.keys())
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # 创建模型
    model = get_model(num_classes=len(class_names))
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 学习率调度器
    if use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 记录最佳验证准确率
    best_val_acc = 0.0
    best_epoch = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        # 学习率调度
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}, 学习率: {current_lr:.6f}")
        
        # 训练阶段
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        
        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        
        # 更新学习率
        if use_scheduler:
            scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_model_and_history(model, history, save_dir, filename_prefix="best_model")
            print(f"保存最佳模型，验证准确率: {best_val_acc:.2f}%")
        
        # 早停
        if use_early_stopping and early_stopping(history['val_loss'], patience=patience):
            print(f"早停触发，停止训练（共训练 {epoch+1} 轮）")
            break
    
    # 训练结束时间
    end_time = time.time()
    training_time = end_time - start_time
    print(f"训练完成，总耗时: {training_time:.2f} 秒")
    print(f"最佳验证准确率: {best_val_acc:.2f}%，轮次: {best_epoch+1}")
    
    # 保存最终模型
    save_model_and_history(model, history, save_dir, filename_prefix="final_model")
    
    # 绘制训练历史
    plot_training_history(history, save_path=os.path.join(save_dir, "training_history.png"))
    
    return model, history, idx_to_class

def main():
    parser = argparse.ArgumentParser(description="训练水稻叶片疾病分类模型")
    parser.add_argument('--data_dir', type=str, default="../rice leaf diseases dataset",
                        help="数据集路径")
    parser.add_argument('--save_dir', type=str, default="results",
                        help="结果保存路径")
    parser.add_argument('--epochs', type=int, default=30,
                        help="训练轮次")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="批处理大小")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="学习率")
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument('--patience', type=int, default=5,
                        help="早停的耐心值")
    parser.add_argument('--no_early_stopping', action='store_true',
                        help="不使用早停策略")
    parser.add_argument('--no_scheduler', action='store_true',
                        help="不使用学习率调度器")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="数据加载的工作线程数")
    
    args = parser.parse_args()
    
    # 训练模型
    train_model(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        use_early_stopping=not args.no_early_stopping,
        use_scheduler=not args.no_scheduler,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main() 