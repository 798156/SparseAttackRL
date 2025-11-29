# test_attack_output.py
"""测试攻击方法的输出，确认它们返回的是什么"""

import torch
import torchvision
import torchvision.transforms as transforms
from target_model import load_target_model
from jsma_attack import jsma_attack
from sparsefool_attack import sparsefool_attack_simple
from evaluation_metrics import compute_l0_norm

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

print("=" * 70)
print("🧪 测试攻击方法的输出和成功率")
print("=" * 70)

# 测试10个样本
jsma_successes = 0
sparsefool_successes = 0

for idx in range(10):
    image, label = testset[idx]
    
    # 检查原始预测
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        pred = output.argmax(dim=1).item()
        
        if pred != label:
            print(f"\n样本 #{idx}: 模型预测错误 (真实{label} vs 预测{pred})，跳过")
            continue
    
    print(f"\n样本 #{idx}: 标签={label}, 原始预测={pred}")
    
    # 测试JSMA
    print(f"  📍 JSMA (theta=1.0):")
    success_jsma, adv_jsma, pixels_jsma = jsma_attack(
        image.to(device), label, model, max_pixels=5, theta=1.0
    )
    
    # 检查对抗样本的预测
    with torch.no_grad():
        output_adv = model(adv_jsma.unsqueeze(0).to(device))
        pred_adv = output_adv.argmax(dim=1).item()
    
    l0_jsma = compute_l0_norm(image.cpu(), adv_jsma.cpu())
    
    print(f"    success返回值: {success_jsma}")
    print(f"    实际L0: {l0_jsma}")
    print(f"    修改像素数: {len(pixels_jsma)}")
    print(f"    对抗预测: {pred_adv}")
    print(f"    真正成功? {pred_adv != label}")
    
    if success_jsma:
        jsma_successes += 1
    
    # 测试SparseFool
    print(f"  📍 SparseFool (perturbation=0.3):")
    success_sf, adv_sf, pixels_sf = sparsefool_attack_simple(
        image.to(device), label, model, max_pixels=5
    )
    
    with torch.no_grad():
        output_adv_sf = model(adv_sf.unsqueeze(0).to(device))
        pred_adv_sf = output_adv_sf.argmax(dim=1).item()
    
    l0_sf = compute_l0_norm(image.cpu(), adv_sf.cpu())
    
    print(f"    success返回值: {success_sf}")
    print(f"    实际L0: {l0_sf}")
    print(f"    修改像素数: {len(pixels_sf)}")
    print(f"    对抗预测: {pred_adv_sf}")
    print(f"    真正成功? {pred_adv_sf != label}")
    
    if success_sf:
        sparsefool_successes += 1

print("\n" + "=" * 70)
print(f"📊 汇总 (10个样本):")
print(f"  JSMA 成功率: {jsma_successes}/10 = {jsma_successes*10}%")
print(f"  SparseFool 成功率: {sparsefool_successes}/10 = {sparsefool_successes*10}%")
print("=" * 70)
