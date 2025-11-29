# download_pretrained_cifar10.py
"""
下载并保存CIFAR-10预训练的ResNet18模型
使用PyTorch Hub或直接训练一个简单模型
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

def main():
    """主训练函数"""
    print("=" * 70)
    print("🚀 训练CIFAR-10 ResNet18模型")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 数据加载
    print("\n📦 加载CIFAR-10数据...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Windows需要num_workers=0避免多进程问题
    num_workers = 0  # Windows上设为0，Linux/Mac可以设为2
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

    # 创建模型
    print("\n🔧 创建ResNet18模型...")
    model = torchvision.models.resnet18(weights=None)  # 不使用预训练
    model.fc = nn.Linear(512, 10)
    model = model.to(device)

    # 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # 快速训练（20 epochs，约15分钟GPU）
    print("\n🎓 开始训练（20 epochs）...")
    print("提示：GPU约15-20分钟，CPU约2-3小时\n")

    num_epochs = 20
    best_acc = 0

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
            
            if batch_idx % 100 == 0:
                print(f'  Epoch {epoch+1}/{num_epochs} [{batch_idx}/{len(trainloader)}] '
                      f'Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.2f}%')
        
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
        
        acc = 100. * correct / total
        print(f'\n✅ Epoch {epoch+1}: Test Acc = {acc:.2f}%')
        
        # 保存最佳模型
        if acc > best_acc:
            print(f'   💾 保存最佳模型 (Acc: {acc:.2f}%)')
            best_acc = acc
            torch.save(model.state_dict(), 'cifar10_resnet18.pth')
        
        scheduler.step()
        print()

    print("=" * 70)
    print(f"🎉 训练完成！最佳准确率: {best_acc:.2f}%")
    print(f"📁 模型保存在: cifar10_resnet18.pth")
    print("=" * 70)


if __name__ == '__main__':
    main()

