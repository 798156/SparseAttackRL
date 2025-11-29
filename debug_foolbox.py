"""
调试Foolbox API的正确用法
"""

import torch
import torchvision
import torchvision.transforms as transforms
from target_model import load_target_model
import foolbox as fb
from foolbox.attacks import FGSM, L2DeepFoolAttack
import numpy as np

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
print("🔍 调试Foolbox API")
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

# 方法1: 尝试不同的预处理设置
print("\n" + "=" * 80)
print("测试不同的预处理设置")
print("=" * 80)

# 测试1: 使用标准化预处理
print("\n📍 测试1: 使用标准化预处理 (mean=0.5, std=0.5)")
try:
    preprocessing = {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'axis': -3
    }
    fmodel = fb.PyTorchModel(model, bounds=(-1, 1), preprocessing=preprocessing)
    
    # FGSM攻击
    attack = FGSM()
    epsilons = [0.1, 0.3, 0.5]
    
    image_np = image.cpu().numpy()
    label_np = np.array([label])
    
    _, advs, success = attack(fmodel, image_np[np.newaxis, ...], label_np, epsilons=epsilons)
    
    print(f"  攻击结果: {success}")
    for i, eps in enumerate(epsilons):
        if success[i]:
            adv = torch.from_numpy(advs[i]).to(device)
            with torch.no_grad():
                final_pred = model(adv.unsqueeze(0)).argmax(dim=1).item()
            print(f"  ✅ epsilon={eps}: 成功! 最终预测={final_pred}")
        else:
            print(f"  ❌ epsilon={eps}: 失败")
except Exception as e:
    print(f"  ❌ 错误: {e}")
    import traceback
    traceback.print_exc()

# 测试2: 不使用预处理
print("\n📍 测试2: 不使用预处理")
try:
    fmodel = fb.PyTorchModel(model, bounds=(-1, 1))
    
    attack = FGSM()
    epsilons = [0.1, 0.3, 0.5]
    
    image_np = image.cpu().numpy()
    label_np = np.array([label])
    
    _, advs, success = attack(fmodel, image_np[np.newaxis, ...], label_np, epsilons=epsilons)
    
    print(f"  攻击结果: {success}")
    for i, eps in enumerate(epsilons):
        if success[i]:
            adv = torch.from_numpy(advs[i]).to(device)
            with torch.no_grad():
                final_pred = model(adv.unsqueeze(0)).argmax(dim=1).item()
            print(f"  ✅ epsilon={eps}: 成功! 最终预测={final_pred}")
        else:
            print(f"  ❌ epsilon={eps}: 失败")
except Exception as e:
    print(f"  ❌ 错误: {e}")
    import traceback
    traceback.print_exc()

# 测试3: DeepFool
print("\n📍 测试3: DeepFool攻击")
try:
    fmodel = fb.PyTorchModel(model, bounds=(-1, 1))
    
    attack = L2DeepFoolAttack()
    
    image_np = image.cpu().numpy()
    label_np = np.array([label])
    
    _, advs, success = attack(fmodel, image_np[np.newaxis, ...], label_np, epsilons=None)
    
    print(f"  攻击结果: {success}")
    if success[0]:
        adv = torch.from_numpy(advs[0]).to(device)
        with torch.no_grad():
            final_pred = model(adv.unsqueeze(0)).argmax(dim=1).item()
        print(f"  ✅ 成功! 最终预测={final_pred}")
        
        # 计算L0
        diff = torch.abs(adv - image)
        modified_mask = diff.sum(dim=0) > 1e-6
        l0 = modified_mask.sum().item()
        print(f"  📊 L0范数: {l0}")
    else:
        print(f"  ❌ 失败")
except Exception as e:
    print(f"  ❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("💡 调试信息")
print("=" * 80)
print("""
常见问题：
1. bounds设置不对 - 应该是(-1, 1)还是(0, 1)？
2. 预处理参数不匹配
3. 模型输入格式不对
4. epsilon值太小

解决方案：
- 如果Foolbox太难集成，可以考虑使用ART (Adversarial Robustness Toolbox)
- 或者继续优化自己的实现，在论文中说明实现细节
""")

