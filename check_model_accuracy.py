# check_model_accuracy.py
"""检查目标模型的准确率"""

import torch
from torchvision import datasets, transforms
from target_model import load_target_model
from tqdm import tqdm

print("=" * 70)
print("🔍 检查目标模型准确率")
print("=" * 70)

# 加载数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 加载模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_target_model().to(device).eval()

print(f"\n📊 测试模型准确率...")
print(f"设备: {device}")
print(f"测试样本数: 1000")

# 测试1000个样本
correct = 0
total = 0

with torch.no_grad():
    for i in tqdm(range(1000), desc="测试进度"):
        image, label = test_set[i]
        image = image.unsqueeze(0).to(device)
        
        output = model(image)
        pred = output.argmax(dim=1).item()
        
        if pred == label:
            correct += 1
        total += 1

accuracy = 100.0 * correct / total

print("\n" + "=" * 70)
print(f"📈 模型准确率: {accuracy:.2f}% ({correct}/{total})")
print("=" * 70)

if accuracy < 70:
    print("\n⚠️  警告：模型准确率低于70%！")
    print("建议：")
    print("1. 使用预训练的模型")
    print("2. 或者训练一个更好的模型")
    print("3. 低准确率会导致很多样本不需要攻击就预测错误")
elif accuracy < 85:
    print("\n⚠️  模型准确率偏低（建议>85%）")
    print("这可能影响攻击实验的有效性")
else:
    print(f"\n✅ 模型准确率良好（{accuracy:.2f}%）")
    print("可以进行对抗攻击实验")

print("=" * 70)

