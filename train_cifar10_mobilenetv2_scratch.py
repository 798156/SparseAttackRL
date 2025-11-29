# train_cifar10_mobilenetv2_scratch.py
"""
从头训练CIFAR-10 MobileNetV2模型（不用预训练权重）
针对CIFAR-10的32x32小图像优化

预计时间：GPU约1-2小时
目标准确率：85-90%
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def main():
    print("=" * 80)
    print("🚀 从头训练CIFAR-10 MobileNetV2（不用预训练）")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    if device == 'cpu':
        print("⚠️  警告：使用CPU训练会非常慢！建议使用GPU")
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            return

    # 数据增强（针对小图像）
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

    # 从头创建MobileNetV2（不用预训练权重）
    print("\n🔧 创建MobileNetV2模型（从头训练）...")
    model = torchvision.models.mobilenet_v2(weights=None)  # 不用预训练权重
    
    # 修改分类器
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    model = model.to(device)
    
    print(f"✅ MobileNetV2创建完成（随机初始化）")
    print(f"   总参数: {sum(p.numel() for p in model.parameters()):,}")

    # 训练设置
    criterion = nn.CrossEntropyLoss()
    
    # 从头训练，所有层用相同的学习率
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,  # 从头训练用较大的学习率
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[60, 120, 160],  # 更长的训练周期
        gamma=0.2
    )

    num_epochs = 200  # 从头训练需要更多epoch
    print(f"\n🎓 开始训练（{num_epochs} epochs）...")
    print(f"预计时间: GPU约1-2小时")
    print(f"目标准确率: 85-90%\n")

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
            torch.save(model.state_dict(), 'cifar10_mobilenetv2.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': test_acc,
            }, 'cifar10_mobilenetv2_best.pth')
        else:
            print()
        
        scheduler.step()

    print("\n" + "=" * 80)
    print(f"🎉 训练完成！最佳准确率: {best_acc:.2f}%")
    print(f"📁 模型保存在: cifar10_mobilenetv2.pth")
    print("=" * 80)


if __name__ == '__main__':
    main()








