# debug_jsma_detailed.py
"""详细调试JSMA攻击，查看每一步的变化"""

import torch
import torchvision
import torchvision.transforms as transforms
from target_model import load_target_model
from jsma_attack import jsma_attack
from evaluation_metrics import compute_l0_norm
import torch.nn.functional as F

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
print("🔬 详细调试JSMA攻击过程")
print("=" * 80)

# 找一个模型预测正确的样本
idx = 0
while idx < 100:
    image, label = testset[idx]
    
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        pred = output.argmax(dim=1).item()
        probs = F.softmax(output, dim=1)[0]
        confidence = probs[pred].item()
        
        if pred == label and confidence > 0.5:
            break
    idx += 1

if idx >= 100:
    print("❌ 找不到合适的测试样本")
    exit(1)

print(f"\n📍 测试样本 #{idx}")
print(f"  真实标签: {label}")
print(f"  模型预测: {pred}")
print(f"  置信度: {confidence:.4f}")

# 手动执行JSMA攻击并记录每一步
print("\n" + "=" * 80)
print("🎯 开始JSMA攻击 (theta=10.0, max_pixels=5)")
print("=" * 80)

adv_image = image.clone().to(device)
adv_image_batch = adv_image.unsqueeze(0)

for step in range(5):
    print(f"\n--- 步骤 {step + 1} ---")
    
    # 检查当前预测
    with torch.no_grad():
        output = model(adv_image_batch)
        pred = output.argmax(dim=1).item()
        probs = F.softmax(output, dim=1)[0]
        confidence = probs[pred].item()
        top5 = torch.topk(probs, 5)
        
        print(f"  当前预测: {pred} (置信度: {confidence:.4f})")
        print(f"  Top-5 预测: {top5.indices.tolist()}")
        print(f"  Top-5 置信度: {[f'{p:.4f}' for p in top5.values.tolist()]}")
        
        if pred != label:
            print(f"  ✅ 攻击成功！预测从 {label} 变为 {pred}")
            break
    
    # 计算梯度
    adv_image_batch.requires_grad_(True)
    output = model(adv_image_batch)
    
    model.zero_grad()
    output[0, label].backward()
    grad = adv_image_batch.grad
    
    # 找到梯度最大的位置
    grad_abs = torch.abs(grad[0])
    max_val = grad_abs.max().item()
    max_idx = grad_abs.argmax().item()
    
    C, H, W = adv_image.shape
    c = max_idx // (H * W)
    h = (max_idx % (H * W)) // W
    w = max_idx % W
    
    print(f"  选中像素: channel={c}, h={h}, w={w}")
    print(f"  梯度值: {grad[0, c, h, w].item():.6f}")
    print(f"  原始值: {adv_image[c, h, w].item():.6f}")
    
    # 应用扰动 (theta=10.0)
    with torch.no_grad():
        direction = -torch.sign(grad[0, c, h, w])
        adv_image[c, h, w] += direction * 10.0
        adv_image_batch = adv_image.unsqueeze(0)
    
    print(f"  修改后: {adv_image[c, h, w].item():.6f}")
    print(f"  变化量: {direction.item() * 10.0:.6f}")

# 最终结果
print("\n" + "=" * 80)
print("📊 最终结果")
print("=" * 80)

with torch.no_grad():
    output = model(adv_image_batch)
    pred_final = output.argmax(dim=1).item()
    probs_final = F.softmax(output, dim=1)[0]
    confidence_final = probs_final[pred_final].item()

l0 = compute_l0_norm(image.cpu(), adv_image.cpu())

print(f"  原始预测: {label}")
print(f"  最终预测: {pred_final}")
print(f"  最终置信度: {confidence_final:.4f}")
print(f"  L0范数: {l0}")
print(f"  攻击结果: {'✅ 成功' if pred_final != label else '❌ 失败'}")

# 使用官方JSMA函数验证
print("\n" + "=" * 80)
print("🔍 使用官方JSMA函数验证")
print("=" * 80)

success_official, adv_official, pixels_official = jsma_attack(
    image.to(device), label, model, max_pixels=5, theta=10.0
)

l0_official = compute_l0_norm(image.cpu(), adv_official.cpu())

with torch.no_grad():
    output_official = model(adv_official.unsqueeze(0).to(device))
    pred_official = output_official.argmax(dim=1).item()

print(f"  Success标志: {success_official}")
print(f"  修改像素数: {len(pixels_official)}")
print(f"  最终预测: {pred_official}")
print(f"  L0范数: {l0_official}")
print(f"  真实成功: {pred_official != label}")
