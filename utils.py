import os
import random
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import cv2
from PIL import Image
import torch.nn.functional as F
import pandas as pd

# 设置随机种子，确保结果可重复
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 创建保存结果的目录
def create_dirs(dirs_list):
    for dir_path in dirs_list:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"创建目录: {dir_path}")

# 保存模型和训练历史
def save_model_and_history(model, history, save_dir, filename_prefix="model"):
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 保存模型
    model_path = os.path.join(save_dir, f"{filename_prefix}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")
    
    # 保存训练历史
    history_path = os.path.join(save_dir, f"{filename_prefix}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"训练历史已保存到 {history_path}")
    
    return model_path, history_path

# 绘制训练历史曲线
def plot_training_history(history, save_path=None):
    epochs = len(history['train_loss'])
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), history['train_loss'], label='训练损失')
    plt.plot(range(1, epochs+1), history['val_loss'], label='验证损失')
    plt.xlabel('周期')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), history['train_acc'], label='训练准确率')
    plt.plot(range(1, epochs+1), history['val_acc'], label='验证准确率')
    plt.xlabel('周期')
    plt.ylabel('准确率')
    plt.title('训练和验证准确率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"训练历史图表已保存到 {save_path}")
    
    plt.show()

# 计算混淆矩阵并绘制
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='混淆矩阵', 
                          cmap=plt.cm.Blues, save_path=None):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("归一化混淆矩阵")
    else:
        print('混淆矩阵，非归一化')
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    if save_path:
        plt.savefig(save_path)
        print(f"混淆矩阵已保存到 {save_path}")
    
    plt.show()
    
    return cm

# 计算并打印分类报告
def print_classification_report(y_true, y_pred, target_names, save_path=None):
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print(report)
    
    # 将分类报告转换为DataFrame并保存
    if save_path:
        report_dict = classification_report(y_true, y_pred, target_names=target_names, digits=4, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()
        df_report.to_csv(save_path)
        print(f"分类报告已保存到 {save_path}")
    
    return report

# 可视化预测结果
def visualize_predictions(model, test_loader, class_names, device, num_images=6, save_dir=None):
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                if images_so_far >= num_images:
                    model.train()
                    if save_dir:
                        save_path = os.path.join(save_dir, 'prediction_samples.png')
                        plt.savefig(save_path)
                        print(f"预测样本可视化已保存到 {save_path}")
                    plt.tight_layout()
                    plt.show()
                    return
                
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'预测: {class_names[preds[j]]}\n真实: {class_names[labels[j]]}',
                            color=("green" if preds[j] == labels[j] else "red"))
                
                # 将张量转换回图像进行显示
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                # 反标准化
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                ax.imshow(img)
                
                if images_so_far == num_images:
                    model.train()
                    if save_dir:
                        save_path = os.path.join(save_dir, 'prediction_samples.png')
                        plt.savefig(save_path)
                        print(f"预测样本可视化已保存到 {save_path}")
                    plt.tight_layout()
                    plt.show()
                    return
        
        model.train()
        if images_so_far > 0:
            if save_dir:
                save_path = os.path.join(save_dir, 'prediction_samples.png')
                plt.savefig(save_path)
                print(f"预测样本可视化已保存到 {save_path}")
            plt.tight_layout()
            plt.show()

# Grad-CAM 可视化
def visualize_model_attention(model, img_path, class_names, transform, target_layer_name='layer4', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    # 加载图像
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 前向传播
    model.zero_grad()
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    pred_class_name = class_names[pred_class]
    
    # 获取特征图
    feature_maps = {}
    gradients = {}
    
    def save_feature_maps(module, input, output):
        feature_maps[target_layer_name] = output.detach()
    
    def save_gradients(module, grad_input, grad_output):
        gradients[target_layer_name] = grad_output[0].detach()
    
    # 注册钩子
    for name, module in model.named_modules():
        if name == target_layer_name:
            module.register_forward_hook(save_feature_maps)
            module.register_backward_hook(save_gradients)
    
    # 计算梯度
    model.zero_grad()
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    pred_score = output[0, pred_class]
    pred_score.backward()
    
    # 获取梯度和特征图
    gradients = gradients[target_layer_name]
    activations = feature_maps[target_layer_name]
    
    # 计算权重
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    
    # 计算加权激活图
    cam = torch.sum(weights * activations, dim=1).squeeze().cpu().numpy()
    
    # 后处理CAM
    cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, img.size)
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    # 可视化原始图像和热力图
    plt.figure(figsize=(12, 5))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(img))
    plt.title(f'预测: {pred_class_name}')
    plt.axis('off')
    
    # 显示热力图叠加图像
    plt.subplot(1, 2, 2)
    img_np = np.array(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(np.uint8(img_np), 0.6, np.uint8(heatmap), 0.4, 0)
    plt.imshow(superimposed_img)
    plt.title('Grad-CAM')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return pred_class, cam

# 调整学习率的辅助函数
def adjust_learning_rate(optimizer, epoch, initial_lr, lr_schedule):
    lr = initial_lr
    if lr_schedule['type'] == 'step':
        for milestone in lr_schedule['steps']:
            if epoch >= milestone:
                lr *= lr_schedule['gamma']
    
    elif lr_schedule['type'] == 'cosine':
        max_epochs = lr_schedule['max_epochs']
        lr = initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))
    
    # 更新优化器中的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

# 判断是否应该提前停止训练
def early_stopping(val_losses, patience=5, delta=0):
    if len(val_losses) <= patience:
        return False
    
    # 检查最后patience轮的验证损失是否都没有改善
    best_loss = min(val_losses[:-patience])
    
    for i in range(patience):
        if val_losses[-(i+1)] < best_loss - delta:
            return False
    
    return True

# 测试时代码
if __name__ == "__main__":
    # 测试创建目录
    create_dirs(['test_dir'])
    
    # 测试绘制训练历史
    history = {
        'train_loss': [0.9, 0.7, 0.5, 0.3, 0.2],
        'val_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
        'train_acc': [0.5, 0.6, 0.7, 0.8, 0.9],
        'val_acc': [0.4, 0.5, 0.6, 0.7, 0.8]
    }
    plot_training_history(history, save_path='test_dir/history.png')
    
    # 测试混淆矩阵绘制
    y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2, 0]
    y_pred = [0, 2, 1, 0, 2, 1, 0, 2, 2, 0]
    class_names = ['Bacterialblight', 'Brownspot', 'Leafsmut']
    plot_confusion_matrix(y_true, y_pred, class_names, save_path='test_dir/confusion_matrix.png')
    
    # 测试分类报告
    print_classification_report(y_true, y_pred, class_names, save_path='test_dir/classification_report.csv')
    
    # 清理测试目录
    import shutil
    shutil.rmtree('test_dir', ignore_errors=True) 