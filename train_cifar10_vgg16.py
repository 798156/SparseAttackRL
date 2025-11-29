# train_cifar10_vgg16.py
"""
训练CIFAR-10 VGG16模型
使用预训练权重微调 + 数据增强

预计时间：GPU约30-40分钟
目标准确率：80-85%
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os


def main():
    print("=" * 80)
    print("🚀 训练CIFAR-10 VGG16 - 预训练微调方式")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    if device == 'cpu':
        print("⚠️  警告：使用CPU训练会非常慢！建议使用GPU")
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            return

    # 数据增强
    print("\n📦 加载CIFAR-10数据...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    num_workers = 0
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

    # 使用ImageNet预训练权重
    print("\n🔧 创建VGG16模型（使用ImageNet预训练权重）...")
    model = torchvision.models.vgg16(weights='IMAGENET1K_V1')
    
    # 修改最后的分类层
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 10)
    model = model.to(device)

    # 训练设置
    criterion = nn.CrossEntropyLoss()
    
    # 使用不同的学习率：预训练层用小学习率，新层用大学习率
    params_pretrained = []
    params_new = []
    
    for name, param in model.named_parameters():
        if 'classifier.6' in name:  # 最后一层
            params_new.append(param)
        else:  # 预训练层
            params_pretrained.append(param)
    
    optimizer = torch.optim.SGD([
        {'params': params_pretrained, 'lr': 0.01},  # 预训练层用小学习率
        {'params': params_new, 'lr': 0.1}  # 新层用大学习率
    ], momentum=0.9, weight_decay=5e-4)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)

    num_epochs = 30  # VGG16需要更多epoch
    print(f"\n🎓 开始训练（{num_epochs} epochs）...")
    print(f"预计时间: GPU约30-40分钟\n")

    best_acc = 0
    save_interval = 10  # 每10个epoch保存一次检查点

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
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
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1:3d}/{num_epochs} | LR: {current_lr:.4f} | '
              f'Train: {train_acc:6.2f}% | Test: {test_acc:6.2f}%', end='')
        
        # 保存最佳模型
        if test_acc > best_acc:
            print(f' 💾 [最佳]')
            best_acc = test_acc
            torch.save(model.state_dict(), 'cifar10_vgg16.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': test_acc,
            }, 'cifar10_vgg16_best.pth')
        else:
            print()
        
        # 定期保存检查点
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f'checkpoint_vgg16_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, checkpoint_path)
            print(f'  💾 检查点保存: {checkpoint_path}')
        
        scheduler.step()

    print("\n" + "=" * 80)
    print(f"🎉 训练完成！最佳准确率: {best_acc:.2f}%")
    print(f"📁 模型保存在: cifar10_vgg16.pth")
    print("=" * 80)


if __name__ == '__main__':
    main()








