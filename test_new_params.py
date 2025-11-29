# test_new_params.py
"""测试新参数的JSMA和SparseFool"""

import torch
from torchvision import datasets, transforms
from target_model import load_target_model
from jsma_attack import jsma_attack
from sparsefool_attack import sparsefool_attack_simple
from evaluation_metrics import compute_l0_norm

print("=" * 70)
print("🔍 测试新参数")
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

print(f"\n测试10个样本...")

jsma_success = 0
sparsefool_success = 0
total = 0

for i in range(20):
    image, label = test_set[i]
    image = image.to(device)
    
    # 检查初始预测
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        initial_pred = output.argmax(dim=1).item()
    
    if initial_pred != label:
        continue
    
    total += 1
    
    # 测试JSMA (theta=10.0)
    success_jsma, adv_img_jsma, pixels_jsma = jsma_attack(
        image, label, model, max_pixels=5, theta=10.0
    )
    if success_jsma:
        l0_jsma = compute_l0_norm(image, adv_img_jsma)
        if l0_jsma > 0:
            jsma_success += 1
    
    # 测试SparseFool (perturbation=0.8)
    success_sf, adv_img_sf, pixels_sf = sparsefool_attack_simple(
        image, label, model, max_pixels=5
    )
    if success_sf:
        l0_sf = compute_l0_norm(image, adv_img_sf)
        if l0_sf > 0:
            sparsefool_success += 1
    
    if total >= 10:
        break

print("\n" + "=" * 70)
print(f"📊 结果 (10个样本):")
print(f"  JSMA (theta=10.0):      {jsma_success}/10 = {100*jsma_success/10:.0f}%")
print(f"  SparseFool (pert=0.8):  {sparsefool_success}/10 = {100*sparsefool_success/10:.0f}%")
print("=" * 70)

if jsma_success > 5 and sparsefool_success > 5:
    print("✅ 参数调整成功！可以运行完整实验了")
else:
    print("⚠️  参数可能还需要进一步调整")

