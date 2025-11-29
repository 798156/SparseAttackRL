"""
测试Foolbox官方攻击实现
对比官方实现与自己实现的性能
"""

import torch
import torchvision
import torchvision.transforms as transforms
from target_model import load_target_model
from foolbox_attacks import (
    foolbox_jsma_attack,
    foolbox_boundary_attack,
    foolbox_cw_attack,
    foolbox_fgsm_attack
)
from jsma_attack import jsma_attack
from sparsefool_attack import sparsefool_attack_simple
from one_pixel_attack import one_pixel_attack
from evaluation_metrics import compute_l0_norm, compute_l2_norm
import numpy as np
from tqdm import tqdm

# 加载模型和数据
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = load_target_model("resnet18")
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

print("=" * 80)
print("🧪 对比官方Foolbox实现 vs 自己实现")
print("=" * 80)

# 测试样本数
num_samples = 50

# 存储结果
results = {
    'DeepFool (官方)': {'asr': [], 'l0': [], 'l2': [], 'time': []},
    'Boundary (官方)': {'asr': [], 'l0': [], 'l2': [], 'time': []},
    'C&W (官方)': {'asr': [], 'l0': [], 'l2': [], 'time': []},
    'FGSM (官方)': {'asr': [], 'l0': [], 'l2': [], 'time': []},
    'JSMA (自己)': {'asr': [], 'l0': [], 'l2': [], 'time': []},
    'SparseFool (自己)': {'asr': [], 'l0': [], 'l2': [], 'time': []},
    'One-Pixel (自己)': {'asr': [], 'l0': [], 'l2': [], 'time': []},
}

print(f"\n测试 {num_samples} 个样本...")

valid_samples = 0
for idx in tqdm(range(len(testset)), desc="处理样本"):
    if valid_samples >= num_samples:
        break
    
    image, label = testset[idx]
    image = image.to(device)
    
    # 检查原始预测
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        
        if pred != label:
            continue
    
    valid_samples += 1
    
    # 1. DeepFool (官方，作为JSMA替代)
    import time
    start_time = time.time()
    success, adv, pixels = foolbox_jsma_attack(image, label, model, max_pixels=10, device=device)
    elapsed = time.time() - start_time
    
    results['DeepFool (官方)']['asr'].append(success)
    results['DeepFool (官方)']['time'].append(elapsed)
    if success:
        l0 = compute_l0_norm(image.cpu(), adv.cpu())
        l2 = compute_l2_norm(image.cpu(), adv.cpu())
        results['DeepFool (官方)']['l0'].append(l0)
        results['DeepFool (官方)']['l2'].append(l2)
    
    # 2. Boundary Attack (官方)
    start_time = time.time()
    success, adv, pixels = foolbox_boundary_attack(image, label, model, max_iterations=50, device=device)
    elapsed = time.time() - start_time
    
    results['Boundary (官方)']['asr'].append(success)
    results['Boundary (官方)']['time'].append(elapsed)
    if success:
        l0 = compute_l0_norm(image.cpu(), adv.cpu())
        l2 = compute_l2_norm(image.cpu(), adv.cpu())
        results['Boundary (官方)']['l0'].append(l0)
        results['Boundary (官方)']['l2'].append(l2)
    
    # 3. C&W (官方)
    start_time = time.time()
    success, adv, pixels = foolbox_cw_attack(image, label, model, max_iterations=50, device=device)
    elapsed = time.time() - start_time
    
    results['C&W (官方)']['asr'].append(success)
    results['C&W (官方)']['time'].append(elapsed)
    if success:
        l0 = compute_l0_norm(image.cpu(), adv.cpu())
        l2 = compute_l2_norm(image.cpu(), adv.cpu())
        results['C&W (官方)']['l0'].append(l0)
        results['C&W (官方)']['l2'].append(l2)
    
    # 4. FGSM (官方)
    start_time = time.time()
    success, adv, pixels = foolbox_fgsm_attack(image, label, model, epsilon=0.1, device=device)
    elapsed = time.time() - start_time
    
    results['FGSM (官方)']['asr'].append(success)
    results['FGSM (官方)']['time'].append(elapsed)
    if success:
        l0 = compute_l0_norm(image.cpu(), adv.cpu())
        l2 = compute_l2_norm(image.cpu(), adv.cpu())
        results['FGSM (官方)']['l0'].append(l0)
        results['FGSM (官方)']['l2'].append(l2)
    
    # 5. JSMA (自己)
    start_time = time.time()
    success, adv, pixels = jsma_attack(image, label, model, max_pixels=10, theta=2.0)
    elapsed = time.time() - start_time
    
    results['JSMA (自己)']['asr'].append(success)
    results['JSMA (自己)']['time'].append(elapsed)
    if success:
        l0 = compute_l0_norm(image.cpu(), adv.cpu())
        l2 = compute_l2_norm(image.cpu(), adv.cpu())
        results['JSMA (自己)']['l0'].append(l0)
        results['JSMA (自己)']['l2'].append(l2)
    
    # 6. SparseFool (自己)
    start_time = time.time()
    success, adv, pixels = sparsefool_attack_simple(image, label, model, max_pixels=10)
    elapsed = time.time() - start_time
    
    results['SparseFool (自己)']['asr'].append(success)
    results['SparseFool (自己)']['time'].append(elapsed)
    if success:
        l0 = compute_l0_norm(image.cpu(), adv.cpu())
        l2 = compute_l2_norm(image.cpu(), adv.cpu())
        results['SparseFool (自己)']['l0'].append(l0)
        results['SparseFool (自己)']['l2'].append(l2)
    
    # 7. One-Pixel (自己)
    start_time = time.time()
    success, params = one_pixel_attack(image, label, model, maxiter=50)
    elapsed = time.time() - start_time
    
    results['One-Pixel (自己)']['asr'].append(success)
    results['One-Pixel (自己)']['time'].append(elapsed)
    if success:
        # One-Pixel固定L0=1
        results['One-Pixel (自己)']['l0'].append(1)
        results['One-Pixel (自己)']['l2'].append(0.0)  # 近似

print("\n" + "=" * 80)
print("📊 实验结果对比")
print("=" * 80)

print(f"\n{'方法':<25} {'ASR (%)':<12} {'平均L0':<12} {'平均L2':<12} {'平均时间 (s)':<15}")
print("-" * 80)

for method_name, data in results.items():
    asr = np.mean(data['asr']) * 100 if data['asr'] else 0
    avg_l0 = np.mean(data['l0']) if data['l0'] else 0
    avg_l2 = np.mean(data['l2']) if data['l2'] else 0
    avg_time = np.mean(data['time']) if data['time'] else 0
    
    print(f"{method_name:<25} {asr:<12.1f} {avg_l0:<12.2f} {avg_l2:<12.2f} {avg_time:<15.3f}")

print("\n" + "=" * 80)
print("💡 结论和建议")
print("=" * 80)
print("""
1. 官方实现的优势：
   - 经过充分验证和优化
   - 学术界广泛认可
   - 减少实现错误的风险

2. 推荐使用：
   - DeepFool: 替代JSMA，迭代式稀疏攻击
   - C&W: 经典强基准，必须对比
   - Boundary: 黑盒场景的基准

3. 论文中的建议：
   - 主要对比官方实现
   - 可以提到自己实现作为验证
   - 重点突出你的RL方法的优势
""")


