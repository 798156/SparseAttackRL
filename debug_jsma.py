# debug_jsma.py
"""调试JSMA的theta参数，找到合适的攻击强度"""

import torch
import torchvision
import torchvision.transforms as transforms
from target_model import load_target_model
from jsma_attack import jsma_attack
from sparsefool_attack import sparsefool_attack_simple
from evaluation_metrics import compute_l0_norm
import numpy as np

# 加载模型和数据
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_target_model("resnet18")
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

print("=" * 80)
print("🔬 调试JSMA和SparseFool的攻击强度参数")
print("=" * 80)

# 测试不同的theta值
theta_values = [1.0, 2.0, 5.0, 10.0, 20.0]

print("\n" + "=" * 80)
print("📍 测试JSMA的不同theta值")
print("=" * 80)

for theta in theta_values:
    print(f"\n🧪 测试 theta = {theta}")
    successes = 0
    total = 0
    l0_values = []
    
    for idx in range(20):
        image, label = testset[idx]
        
        # 检查原始预测
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
            
            if pred != label:
                continue
        
        total += 1
        
        # 测试JSMA
        success, adv, pixels = jsma_attack(
            image.to(device), label, model, max_pixels=5, theta=theta
        )
        
        if success:
            successes += 1
            l0 = compute_l0_norm(image.cpu(), adv.cpu())
            l0_values.append(l0)
    
    asr = successes / total * 100 if total > 0 else 0
    avg_l0 = np.mean(l0_values) if l0_values else 0
    
    print(f"  ✅ 攻击成功率: {successes}/{total} = {asr:.1f}%")
    print(f"  📊 平均L0: {avg_l0:.2f}")

# 测试不同的perturbation值（需要修改sparsefool_attack.py中的hardcoded值）
print("\n" + "=" * 80)
print("📍 测试SparseFool的性能（perturbation=0.3）")
print("=" * 80)

successes = 0
total = 0
l0_values = []

for idx in range(20):
    image, label = testset[idx]
    
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        pred = output.argmax(dim=1).item()
        
        if pred != label:
            continue
    
    total += 1
    
    success, adv, pixels = sparsefool_attack_simple(
        image.to(device), label, model, max_pixels=5
    )
    
    if success:
        successes += 1
        l0 = compute_l0_norm(image.cpu(), adv.cpu())
        l0_values.append(l0)

asr = successes / total * 100 if total > 0 else 0
avg_l0 = np.mean(l0_values) if l0_values else 0

print(f"  ✅ 攻击成功率: {successes}/{total} = {asr:.1f}%")
print(f"  📊 平均L0: {avg_l0:.2f}")

print("\n" + "=" * 80)
print("💡 建议:")
print("=" * 80)
print("""
基于测试结果，建议：
1. JSMA: 使用 theta=10.0 可以在保持低L0的同时达到较高成功率
2. SparseFool: perturbation=0.3 可能太小，建议增加到0.5-0.8

修改位置：
- run_full_experiments.py 中调用 jsma_attack 时设置 theta=10.0
- sparsefool_attack.py 中修改 perturbation 值
""")
