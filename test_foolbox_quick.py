"""
快速测试Foolbox是否正常工作
只测试少量样本
"""

import torch
import torchvision
import torchvision.transforms as transforms
from target_model import load_target_model
from foolbox_attacks import foolbox_jsma_attack, foolbox_fgsm_attack
from evaluation_metrics import compute_l0_norm

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
print("🧪 快速测试Foolbox官方实现")
print("=" * 80)

# 测试5个样本
num_samples = 5
tested = 0

for idx in range(len(testset)):
    if tested >= num_samples:
        break
    
    image, label = testset[idx]
    image = image.to(device)
    
    # 检查原始预测
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        
        if pred != label:
            continue
    
    tested += 1
    print(f"\n{'='*60}")
    print(f"样本 #{tested}")
    print(f"{'='*60}")
    print(f"原始标签: {label}, 模型预测: {pred}")
    
    # 测试DeepFool (作为JSMA替代)
    print("\n🔍 测试 DeepFool (官方)...")
    success, adv, pixels = foolbox_jsma_attack(image, label, model, device=device)
    
    if success:
        l0 = compute_l0_norm(image.cpu(), adv.cpu())
        with torch.no_grad():
            final_pred = model(adv.unsqueeze(0)).argmax(dim=1).item()
        print(f"  ✅ 攻击成功")
        print(f"  📊 L0范数: {l0}")
        print(f"  🎯 最终预测: {final_pred}")
        print(f"  📍 修改像素数: {len(pixels)}")
    else:
        print(f"  ❌ 攻击失败")
    
    # 测试FGSM (官方)
    print("\n🔍 测试 FGSM (官方)...")
    success, adv, pixels = foolbox_fgsm_attack(image, label, model, epsilon=0.1, device=device)
    
    if success:
        l0 = compute_l0_norm(image.cpu(), adv.cpu())
        with torch.no_grad():
            final_pred = model(adv.unsqueeze(0)).argmax(dim=1).item()
        print(f"  ✅ 攻击成功")
        print(f"  📊 L0范数: {l0}")
        print(f"  🎯 最终预测: {final_pred}")
        print(f"  📍 修改像素数: {len(pixels)}")
    else:
        print(f"  ❌ 攻击失败")

print("\n" + "=" * 80)
print("✅ 测试完成！")
print("=" * 80)
print("""
如果看到攻击成功，说明Foolbox集成正常工作！

下一步：
1. 运行 test_foolbox_attacks.py 进行完整对比
2. 更新 run_full_experiments.py 使用官方实现
3. 保留自己实现作为备份
""")


