# debug_jsma_saliency.py
"""调试JSMA显著性图"""

import torch
from torchvision import datasets, transforms
from target_model import load_target_model
from jsma_attack import compute_jacobian, compute_saliency_map

print("=" * 70)
print("🔍 调试JSMA显著性图")
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

# 测试一个样本
image, label = test_set[0]
image = image.to(device).unsqueeze(0)

print(f"\n📊 样本信息:")
print(f"  Label: {label}")

# 初始预测
with torch.no_grad():
    output = model(image)
    pred = output.argmax(dim=1).item()
    probs = torch.softmax(output, dim=1)[0]
    print(f"  初始预测: {pred}, 置信度: {probs[pred]:.4f}")

# 如果预测正确，选择第二高的类别作为目标
if pred == label:
    sorted_preds = output.argsort(dim=1, descending=True)[0]
    target_class = sorted_preds[1].item()
    print(f"  目标类别: {target_class}, 置信度: {probs[target_class]:.4f}")
else:
    target_class = pred
    print(f"  预测已错，目标类别: {target_class}")

# 计算雅可比矩阵
image.requires_grad = True
output = model(image)
jacobian = compute_jacobian(model, image, output)

print(f"\n📈 雅可比矩阵:")
print(f"  形状: {jacobian.shape}")
print(f"  目标类别梯度范围: [{jacobian[0, target_class].min():.4f}, {jacobian[0, target_class].max():.4f}]")
print(f"  源类别梯度范围: [{jacobian[0, label].min():.4f}, {jacobian[0, label].max():.4f}]")

# 计算显著性图
mask = torch.ones((3, 32, 32), dtype=torch.bool, device=device)
saliency_map = compute_saliency_map(jacobian, label, target_class, mask, increase=True)

print(f"\n🎯 显著性图:")
print(f"  形状: {saliency_map.shape}")
print(f"  最大值: {saliency_map.max().item():.6f}")
print(f"  非零元素数: {(saliency_map > 0).sum().item()}")
print(f"  非零元素比例: {100*(saliency_map > 0).sum().item() / saliency_map.numel():.2f}%")

if saliency_map.max() == 0:
    print(f"\n❌ 问题：显著性图全为0！")
    print(f"   可能原因：找不到同时满足条件的像素")
    print(f"   (grad_target > 0 且 grad_source < 0)")
    
    # 检查每个条件
    grad_target = jacobian[0, target_class]
    grad_source = jacobian[0, label]
    
    target_pos = (grad_target > 0).sum().item()
    source_neg = (grad_source < 0).sum().item()
    both = ((grad_target > 0) & (grad_source < 0)).sum().item()
    
    print(f"\n  统计:")
    print(f"    grad_target > 0: {target_pos} 像素 ({100*target_pos/grad_target.numel():.1f}%)")
    print(f"    grad_source < 0: {source_neg} 像素 ({100*source_neg/grad_source.numel():.1f}%)")
    print(f"    两者同时满足: {both} 像素 ({100*both/grad_target.numel():.1f}%)")
    
    if both == 0:
        print(f"\n  ⚠️  没有像素同时满足两个条件！")
        print(f"     这可能是JSMA失败的原因")
        print(f"     解决方案：放宽条件或使用其他策略")
else:
    print(f"\n✅ 找到有效显著性像素")
    
    # 找到最大值位置
    flat_idx = saliency_map.argmax().item()
    C, H, W = 3, 32, 32
    c = flat_idx // (H * W)
    h = (flat_idx % (H * W)) // W
    w = flat_idx % W
    
    print(f"  最大显著性位置: 通道={c}, 行={h}, 列={w}")
    print(f"  显著性值: {saliency_map[flat_idx].item():.6f}")
    print(f"  目标类别梯度: {jacobian[0, target_class, c, h, w].item():.6f}")
    print(f"  源类别梯度: {jacobian[0, label, c, h, w].item():.6f}")

print("\n" + "=" * 70)

