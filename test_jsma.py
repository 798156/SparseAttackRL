# test_jsma.py
"""
测试 JSMA 攻击的简单脚本
"""
import torch
import numpy as np
from torchvision import datasets, transforms
from target_model import load_target_model
from jsma_attack import jsma_attack

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 使用设备: {device}")

# 加载 CIFAR-10 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 加载目标模型
print("🔧 加载目标模型...")
model = load_target_model('resnet18', num_classes=10)
model = model.eval().to(device)

# 测试几个样本
print("\n" + "=" * 50)
print("🧪 开始测试 JSMA Attack")
print("=" * 50)

num_test_samples = 5
success_count = 0

for i in range(num_test_samples):
    image, label = test_set[i]
    
    # 测试原始图像的预测
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        pred = output.argmax(dim=1).item()
    
    print(f"\n样本 {i+1}:")
    print(f"  真实标签: {label}, 模型预测: {pred}")
    
    if pred != label:
        print(f"  ⚠️ 模型预测错误，跳过")
        continue
    
    # 执行 JSMA 攻击
    success, adv_img, modified_pixels = jsma_attack(
        image, label, model, max_pixels=5, theta=1.0
    )
    
    if success:
        # 验证对抗样本
        with torch.no_grad():
            adv_output = model(adv_img.unsqueeze(0).to(device))
            adv_pred = adv_output.argmax(dim=1).item()
        
        print(f"  ✅ 攻击成功！")
        print(f"     修改像素数: {len(modified_pixels)}")
        print(f"     对抗预测: {adv_pred}")
        print(f"     前3个修改的像素: {modified_pixels[:3]}")
        success_count += 1
    else:
        print(f"  ❌ 攻击失败")

print("\n" + "=" * 50)
print(f"✅ 测试完成！成功率: {success_count}/{num_test_samples} = {success_count/num_test_samples*100:.1f}%")
print("=" * 50)

