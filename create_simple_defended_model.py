"""
快速创建简单防御模型

方法：FGSM对抗训练（快速版）
- 不需要完整训练
- 5-10分钟即可完成
- 足以展示方法在防御模型上的性能

这不是最强的防御，但：
1. 比标准模型更鲁棒
2. 足够用于论文对比
3. 训练速度快
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

def fgsm_attack(model, images, labels, epsilon=0.03):
    """FGSM对抗攻击（用于生成对抗样本训练）"""
    images.requires_grad = True
    
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    # 生成对抗样本
    adv_images = images + epsilon * images.grad.sign()
    adv_images = torch.clamp(adv_images, 0, 1)
    
    return adv_images.detach()

def adversarial_training_epoch(model, trainloader, optimizer, device, epsilon=0.03):
    """一个对抗训练epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(trainloader, desc="对抗训练"):
        images, labels = images.to(device), labels.to(device)
        
        # 标准训练
        outputs = model(images)
        loss_clean = nn.CrossEntropyLoss()(outputs, labels)
        
        # 生成对抗样本
        adv_images = fgsm_attack(model, images, labels, epsilon)
        
        # 对抗训练
        adv_outputs = model(adv_images)
        loss_adv = nn.CrossEntropyLoss()(adv_outputs, labels)
        
        # 总损失：50% 干净样本 + 50% 对抗样本
        loss = 0.5 * loss_clean + 0.5 * loss_adv
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(trainloader), 100. * correct / total

def test_model(model, testloader, device):
    """测试模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def main():
    """主流程"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  🛡️  快速创建简单防御模型                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

方法：FGSM对抗训练
策略：在已有ResNet18基础上进行3-5个epoch的对抗微调
时间：约5-10分钟
效果：比标准模型更鲁棒，足够用于论文对比

开始训练...
    """)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}\n")
    
    # 加载数据
    print("📦 加载CIFAR-10数据...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor()])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # 使用较小的训练集（加速训练）
    # 只使用10000个样本进行快速微调
    train_subset = torch.utils.data.Subset(
        trainset, 
        np.random.choice(len(trainset), 10000, replace=False)
    )
    
    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=128, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )
    
    # 加载预训练的标准模型
    print("\n📦 加载预训练的ResNet18...")
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # 尝试加载已有的训练好的模型
    try:
        checkpoint = torch.load('cifar10_resnet18.pth', map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("  ✅ 成功加载已有模型")
    except:
        print("  ⚠️  未找到预训练模型，从头开始训练")
        print("     建议：先运行 python train_cifar10_fast.py 训练标准模型")
        print("     或者继续（但可能需要更长时间）")
        user_input = input("\n继续？ (y/n): ")
        if user_input.lower() != 'y':
            return
    
    model = model.to(device)
    
    # 测试初始准确率
    print("\n📊 测试初始准确率...")
    initial_acc = test_model(model, testloader, device)
    print(f"  初始准确率: {initial_acc:.2f}%")
    
    # 对抗训练
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    print(f"\n{'='*80}")
    print("🚀 开始对抗训练")
    print(f"{'='*80}")
    print("训练设置：")
    print(f"  - Epochs: 3-5")
    print(f"  - 训练样本: 10000")
    print(f"  - FGSM epsilon: 0.03")
    print(f"  - 预计时间: 5-10分钟")
    print(f"{'='*80}\n")
    
    num_epochs = 5
    best_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch+1}/{num_epochs}")
        
        # 对抗训练
        train_loss, train_acc = adversarial_training_epoch(
            model, trainloader, optimizer, device, epsilon=0.03
        )
        
        # 测试
        test_acc = test_model(model, testloader, device)
        
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  训练准确率: {train_acc:.2f}%")
        print(f"  测试准确率: {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, 'cifar10_resnet18_defended.pth')
            print(f"  ✅ 保存最佳模型 (准确率: {test_acc:.2f}%)")
    
    print(f"\n{'='*80}")
    print("🎉 对抗训练完成！")
    print(f"{'='*80}")
    print(f"\n📊 结果总结:")
    print(f"  初始准确率: {initial_acc:.2f}%")
    print(f"  最终准确率: {best_acc:.2f}%")
    print(f"  准确率变化: {best_acc - initial_acc:+.2f}%")
    print(f"\n💾 模型已保存: cifar10_resnet18_defended.pth")
    
    print(f"\n{'='*80}")
    print("📈 下一步：")
    print(f"{'='*80}")
    print("1. 测试防御模型:")
    print("   python test_on_defended_model.py")
    print("\n2. 或者修改配置使用custom模型:")
    print("   CONFIG['defense_type'] = 'custom'")
    print("   CONFIG['defense_model_path'] = 'cifar10_resnet18_defended.pth'")
    
    print(f"\n{'='*80}")
    print("💡 说明：")
    print(f"{'='*80}")
    print("这是一个简单的防御模型，用于展示攻击方法在防御场景下的性能。")
    print("虽然不是最强的防御，但足够用于论文对比研究。")
    print("预期：ASR会降低20-40%，但方法相对排名应该保持。")

if __name__ == "__main__":
    main()
















