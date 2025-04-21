import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 自定义数据集类
class RiceLeafDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 加载数据集
def load_dataset(data_dir):
    img_paths = []
    labels = []
    class_to_idx = {
        'Bacterialblight': 0,
        'Brownspot': 1,
        'Leafsmut': 2
    }
    
    # 遍历所有类别文件夹
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            class_idx = class_to_idx.get(class_name)
            if class_idx is not None:  # 确保只处理我们关心的类别
                # 遍历该类别下的所有图像
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        img_paths.append(img_path)
                        labels.append(class_idx)
    
    return img_paths, labels, class_to_idx

# 划分数据集为训练集、验证集和测试集
def split_dataset(img_paths, labels, test_size=0.15, val_size=0.15, random_state=42):
    # 首先分离出测试集
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        img_paths, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    
    # 从剩余数据中分离出验证集
    val_ratio = val_size / (1 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_ratio, 
        stratify=train_val_labels, random_state=random_state
    )
    
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels

# 创建数据转换
def get_transforms(phase):
    # ImageNet平均值和标准差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

# 创建数据加载器
def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    # 加载数据集
    img_paths, labels, class_to_idx = load_dataset(data_dir)
    
    # 划分数据集
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_dataset(
        img_paths, labels
    )
    
    # 创建数据集实例
    train_dataset = RiceLeafDataset(
        train_paths, train_labels, transform=get_transforms('train')
    )
    val_dataset = RiceLeafDataset(
        val_paths, val_labels, transform=get_transforms('val')
    )
    test_dataset = RiceLeafDataset(
        test_paths, test_labels, transform=get_transforms('test')
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_to_idx

# 用于数据集探索的辅助函数
def get_class_distribution(labels):
    class_counts = {}
    for label in labels:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    return class_counts

# 如果直接运行此脚本，则执行一些测试代码
if __name__ == "__main__":
    set_seed()
    data_dir = "../rice leaf diseases dataset"
    
    img_paths, labels, class_to_idx = load_dataset(data_dir)
    print(f"类别映射: {class_to_idx}")
    
    class_distribution = get_class_distribution(labels)
    print(f"类别分布: {class_distribution}")
    
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = split_dataset(
        img_paths, labels
    )
    
    print(f"训练集大小: {len(train_paths)}")
    print(f"验证集大小: {len(val_paths)}")
    print(f"测试集大小: {len(test_paths)}")
    
    # 测试数据加载器
    train_loader, val_loader, test_loader, _ = create_dataloaders(data_dir, batch_size=4)
    
    # 获取并显示一批数据
    images, labels = next(iter(train_loader))
    print(f"批次形状: {images.shape}")
    print(f"标签: {labels}") 