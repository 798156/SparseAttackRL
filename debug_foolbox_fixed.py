"""
调试Foolbox API的正确用法 - 使用torch tensor
"""

import torch
import torchvision
import torchvision.transforms as transforms
from target_model import load_target_model
import foolbox as fb
from foolbox.attacks import FGSM, L2DeepFoolAttack, PGD

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
print("🔍 调试Foolbox API - 使用torch tensor")
print("=" * 80)

# 获取一个测试样本
for idx in range(100):
    image, label = testset[idx]
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        
        if pred == label:
            break

print(f"\n测试样本: 标签={label}, 预测={pred}")
print(f"图像形状: {image.shape}")
print(f"图像范围: [{image.min():.3f}, {image.max():.3f}]")

# 创建Foolbox模型 - 关键：使用torch tensor输入
print("\n" + "=" * 80)
print("创建Foolbox模型")
print("=" * 80)

# 由于图像已经标准化到[-1, 1]，所以bounds应该是(-1, 1)
fmodel = fb.PyTorchModel(model, bounds=(-1, 1))
print(f"✅ Foolbox模型创建成功")

# 测试1: FGSM攻击
print("\n" + "=" * 80)
print("📍 测试1: FGSM攻击")
print("=" * 80)

try:
    attack = FGSM()
    epsilons = [0.1, 0.3, 0.5]
    
    # 使用torch tensor，不是numpy！
    image_batch = image.unsqueeze(0)  # [1, 3, 32, 32]
    label_tensor = torch.tensor([label]).to(device)
    
    _, advs, success = attack(fmodel, image_batch, label_tensor, epsilons=epsilons)
    
    print(f"  攻击结果: {success}")
    for i, eps in enumerate(epsilons):
        if success[i]:
            adv = advs[i]
            # 确保adv是3D的 [C, H, W]
            if adv.dim() == 4:  # [1, C, H, W]
                adv = adv.squeeze(0)
            
            with torch.no_grad():
                final_pred = model(adv.unsqueeze(0)).argmax(dim=1).item()
            
            # 计算L0
            diff = torch.abs(adv - image)
            modified_mask = diff.sum(dim=0) > 1e-6
            l0 = modified_mask.sum().item()
            
            print(f"  ✅ epsilon={eps}: 成功! 最终预测={final_pred}, L0={l0}")
        else:
            print(f"  ❌ epsilon={eps}: 失败")
except Exception as e:
    print(f"  ❌ 错误: {e}")
    import traceback
    traceback.print_exc()

# 测试2: DeepFool攻击
print("\n" + "=" * 80)
print("📍 测试2: L2 DeepFool攻击")
print("=" * 80)

try:
    attack = L2DeepFoolAttack()
    
    image_batch = image.unsqueeze(0)
    label_tensor = torch.tensor([label]).to(device)
    
    _, advs, success = attack(fmodel, image_batch, label_tensor, epsilons=None)
    
    print(f"  攻击结果: {success}")
    if success[0]:
        adv = advs[0]
        with torch.no_grad():
            final_pred = model(adv.unsqueeze(0)).argmax(dim=1).item()
        
        # 计算L0和L2
        diff = torch.abs(adv - image)
        modified_mask = diff.sum(dim=0) > 1e-6
        l0 = modified_mask.sum().item()
        l2 = torch.norm(diff).item()
        
        print(f"  ✅ 成功! 最终预测={final_pred}")
        print(f"  📊 L0范数: {l0}")
        print(f"  📊 L2范数: {l2:.4f}")
    else:
        print(f"  ❌ 失败")
except Exception as e:
    print(f"  ❌ 错误: {e}")
    import traceback
    traceback.print_exc()

# 测试3: PGD攻击
print("\n" + "=" * 80)
print("📍 测试3: PGD攻击")
print("=" * 80)

try:
    attack = PGD()
    epsilons = [0.1, 0.3]
    
    image_batch = image.unsqueeze(0)
    label_tensor = torch.tensor([label]).to(device)
    
    _, advs, success = attack(fmodel, image_batch, label_tensor, epsilons=epsilons)
    
    print(f"  攻击结果: {success}")
    for i, eps in enumerate(epsilons):
        if success[i]:
            adv = advs[i]
            # 确保adv是3D的 [C, H, W]
            if adv.dim() == 4:  # [1, C, H, W]
                adv = adv.squeeze(0)
            
            with torch.no_grad():
                final_pred = model(adv.unsqueeze(0)).argmax(dim=1).item()
            
            diff = torch.abs(adv - image)
            modified_mask = diff.sum(dim=0) > 1e-6
            l0 = modified_mask.sum().item()
            
            print(f"  ✅ epsilon={eps}: 成功! 最终预测={final_pred}, L0={l0}")
        else:
            print(f"  ❌ epsilon={eps}: 失败")
except Exception as e:
    print(f"  ❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("💡 总结")
print("=" * 80)
print("""
关键发现：
1. Foolbox需要torch tensor输入，不能用numpy array
2. bounds应该设置为(-1, 1)，匹配图像的标准化范围
3. 成功的话，可以集成到实验框架中

下一步：
- 如果以上测试成功，更新foolbox_attacks.py
- 集成到run_full_experiments.py
- 与自己的实现进行对比
""")

