# dataset_loader.py
"""
多数据集加载器
支持：CIFAR-10, CIFAR-100, ImageNet (子集)
"""

import torch
from torchvision import datasets, transforms
import os


class DatasetLoader:
    """统一的数据集加载接口"""
    
    def __init__(self, dataset_name='cifar10', data_root='./data'):
        """
        参数:
            dataset_name: 'cifar10', 'cifar100', 'imagenet'
            data_root: 数据存储路径
        """
        self.dataset_name = dataset_name.lower()
        self.data_root = data_root
        
        # 根据数据集设置参数
        if self.dataset_name == 'cifar10':
            self.num_classes = 10
            self.image_size = 32
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2023, 0.1994, 0.2010]
        elif self.dataset_name == 'cifar100':
            self.num_classes = 100
            self.image_size = 32
            self.mean = [0.5071, 0.4867, 0.4408]
            self.std = [0.2675, 0.2565, 0.2761]
        elif self.dataset_name == 'imagenet':
            self.num_classes = 1000
            self.image_size = 224
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def get_transform(self, train=False):
        """获取数据变换"""
        if train:
            # 训练时可以添加数据增强
            transform_list = [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]
        else:
            # 测试时只做归一化
            if self.dataset_name == 'imagenet':
                transform_list = [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ]
            else:
                transform_list = [
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ]
        
        return transforms.Compose(transform_list)
    
    def load_test_set(self):
        """加载测试集"""
        transform = self.get_transform(train=False)
        
        if self.dataset_name == 'cifar10':
            test_set = datasets.CIFAR10(
                root=self.data_root,
                train=False,
                download=True,
                transform=transform
            )
        
        elif self.dataset_name == 'cifar100':
            test_set = datasets.CIFAR100(
                root=self.data_root,
                train=False,
                download=True,
                transform=transform
            )
        
        elif self.dataset_name == 'imagenet':
            # ImageNet需要手动下载
            # 这里使用验证集的一个子集
            imagenet_path = os.path.join(self.data_root, 'imagenet', 'val')
            if not os.path.exists(imagenet_path):
                print(f"⚠️ ImageNet path not found: {imagenet_path}")
                print("   请手动下载ImageNet验证集并解压到该路径")
                print("   或者跳过ImageNet实验")
                return None
            
            test_set = datasets.ImageFolder(
                root=imagenet_path,
                transform=transform
            )
        
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        return test_set
    
    def get_sample_subset(self, test_set, num_samples=500, seed=42):
        """
        从测试集中随机采样子集
        
        参数:
            test_set: 测试集
            num_samples: 采样数量
            seed: 随机种子
        
        返回:
            indices: 采样的索引列表
        """
        import numpy as np
        np.random.seed(seed)
        
        total_samples = len(test_set)
        num_samples = min(num_samples, total_samples)
        
        # 随机采样
        indices = np.random.choice(total_samples, num_samples, replace=False)
        
        return indices.tolist()
    
    def get_dataset_info(self):
        """获取数据集信息"""
        return {
            'name': self.dataset_name,
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'mean': self.mean,
            'std': self.std
        }


def get_all_datasets(data_root='./data'):
    """
    获取所有可用的数据集
    
    返回:
        datasets_dict: {dataset_name: DatasetLoader}
    """
    datasets_dict = {}
    
    # CIFAR-10（总是可用）
    datasets_dict['cifar10'] = DatasetLoader('cifar10', data_root)
    
    # CIFAR-100（总是可用）
    datasets_dict['cifar100'] = DatasetLoader('cifar100', data_root)
    
    # ImageNet（如果存在）
    imagenet_loader = DatasetLoader('imagenet', data_root)
    imagenet_path = os.path.join(data_root, 'imagenet', 'val')
    if os.path.exists(imagenet_path):
        datasets_dict['imagenet'] = imagenet_loader
    else:
        print("⚠️ ImageNet not found, skipping ImageNet experiments")
    
    return datasets_dict


# 使用示例
if __name__ == "__main__":
    print("🧪 测试数据集加载器")
    
    # 测试CIFAR-10
    loader = DatasetLoader('cifar10')
    print(f"\n数据集: {loader.get_dataset_info()}")
    
    test_set = loader.load_test_set()
    print(f"测试集大小: {len(test_set)}")
    
    # 采样子集
    indices = loader.get_sample_subset(test_set, num_samples=100)
    print(f"采样索引数量: {len(indices)}")
    
    # 测试CIFAR-100
    loader100 = DatasetLoader('cifar100')
    print(f"\n数据集: {loader100.get_dataset_info()}")
    test_set100 = loader100.load_test_set()
    print(f"测试集大小: {len(test_set100)}")
    
    print("\n✅ 测试完成！")

