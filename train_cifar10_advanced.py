# train_cifar10_advanced.py
"""
高级CIFAR-10 ResNet18训练脚本
目标: 达到90%+准确率

优化策略：
1. 更多训练轮数（200 epochs）
2. 更强的数据增强（AutoAugment, Cutout）
3. 学习率Warmup
4. Label Smoothing
5. Mixup数据增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os


class Cutout:
    """随机遮挡正方形区域"""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing交叉熵"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def main():
    """主训练函数"""
    print("=" * 80)
    print("🚀 高级训练CIFAR-10 ResNet18模型 - 目标90%+准确率")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 超参数配置
    config = {
        'num_epochs': 200,
        'batch_size': 128,
        'lr': 0.1,
        'weight_decay': 5e-4,
        'momentum': 0.9,
        'warmup_epochs': 5,
        'use_mixup': True,
        'mixup_alpha': 1.0,
        'label_smoothing': 0.1,
        'use_cutout': True,
    }

    print("\n📋 训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 数据增强 - 更强的增强
    print("\n📦 加载CIFAR-10数据...")
    transform_train_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    
    if config['use_cutout']:
        transform_train_list.append(Cutout(n_holes=1, length=16))
    
    transform_train = transforms.Compose(transform_train_list)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Windows需要num_workers=0避免多进程问题
    num_workers = 0
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

    # 创建模型
    print("\n🔧 创建ResNet18模型...")
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    model = model.to(device)

    # 训练设置
    if config['label_smoothing'] > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing'])
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], 
                                momentum=config['momentum'], 
                                weight_decay=config['weight_decay'])
    
    # 学习率调度器 - MultiStepLR更稳定
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    print(f"\n🎓 开始训练（{config['num_epochs']} epochs）...")
    print(f"预计时间: GPU约2-3小时，CPU约1-2天\n")

    best_acc = 0
    train_acc_history = []
    test_acc_history = []

    for epoch in range(config['num_epochs']):
        # Warmup学习率
        if epoch < config['warmup_epochs']:
            lr = config['lr'] * (epoch + 1) / config['warmup_epochs']
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # 训练
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Mixup
            if config['use_mixup'] and np.random.rand() > 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, 
                                                                 config['mixup_alpha'], device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        train_acc_history.append(train_acc)
        
        # 测试
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        test_acc_history.append(test_acc)
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch+1:3d}/{config["num_epochs"]} | LR: {current_lr:.4f} | '
              f'Train Acc: {train_acc:6.2f}% | Test Acc: {test_acc:6.2f}%', end='')
        
        # 保存最佳模型
        if test_acc > best_acc:
            print(f' 💾 [最佳: {test_acc:.2f}%]')
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
                'config': config
            }, 'cifar10_resnet18_best.pth')
            
            # 同时保存为简单格式（兼容旧代码）
            torch.save(model.state_dict(), 'cifar10_resnet18.pth')
        else:
            print()
        
        # 更新学习率（warmup之后）
        if epoch >= config['warmup_epochs']:
            scheduler.step()
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
                'train_acc_history': train_acc_history,
                'test_acc_history': test_acc_history,
                'config': config
            }, f'checkpoint_epoch_{epoch+1}.pth')

    print("\n" + "=" * 80)
    print(f"🎉 训练完成！最佳准确率: {best_acc:.2f}%")
    print(f"📁 最佳模型保存在: cifar10_resnet18_best.pth")
    print(f"📁 兼容模型保存在: cifar10_resnet18.pth")
    print("=" * 80)
    
    # 显示历史趋势
    print(f"\n📊 最后10个epoch的准确率:")
    for i in range(max(0, len(test_acc_history)-10), len(test_acc_history)):
        print(f"  Epoch {i+1}: {test_acc_history[i]:.2f}%")


if __name__ == '__main__':
    main()




















