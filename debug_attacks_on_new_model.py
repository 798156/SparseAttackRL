"""
调试新模型上的攻击方法
找出为什么所有攻击成功率都很低
"""

import torch
import torchvision
import torchvision.transforms as transforms
from target_model import load_target_model
from jsma_attack import jsma_attack
from one_pixel_attack import one_pixel_attack
from sparsefool_attack import sparsefool_attack_simple
from evaluation_metrics import compute_l0_norm

def debug_single_sample():
    """详细调试单个样本"""
    print("=" * 80)
    print("🔍 调试新模型(88.8%)上的攻击方法")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载模型
    print("\n📦 加载模型...")
    model = load_target_model("resnet18")
    model = model.to(device)
    model.eval()
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # 找一个正确分类的样本
    print("\n📍 寻找正确分类的样本...")
    for idx in range(len(testset)):
        image, label = testset[idx]
        image = image.to(device)
        
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            pred = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, pred].item()
            
            if pred == label:
                print(f"\n✅ 找到样本 #{idx}")
                print(f"   标签: {label}")
                print(f"   预测: {pred}")
                print(f"   置信度: {confidence:.4f}")
                break
    
    # 测试不同强度的攻击
    print("\n" + "=" * 80)
    print("🧪 测试不同攻击强度")
    print("=" * 80)
    
    # 1. One-Pixel - 测试不同max_iter
    print("\n📍 One-Pixel攻击 - 测试不同迭代次数")
    print("-" * 80)
    for max_iter in [50, 100, 200, 400]:
        try:
            success, params = one_pixel_attack(image, label, model, max_iter=max_iter)
            print(f"  max_iter={max_iter:3d}: {'✅ 成功' if success else '❌ 失败'}")
            if success:
                break
        except Exception as e:
            print(f"  max_iter={max_iter:3d}: ❌ 错误 - {e}")
    
    # 2. JSMA - 测试不同theta
    print("\n📍 JSMA攻击 - 测试不同theta值")
    print("-" * 80)
    for theta in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        try:
            success, adv, pixels = jsma_attack(image, label, model, max_pixels=10, theta=theta)
            if success:
                l0 = compute_l0_norm(image.cpu(), adv.cpu())
                with torch.no_grad():
                    final_pred = model(adv.unsqueeze(0)).argmax(dim=1).item()
                    final_conf = torch.softmax(model(adv.unsqueeze(0)), dim=1)[0, final_pred].item()
                print(f"  theta={theta:5.1f}: ✅ 成功, L0={l0}, 最终预测={final_pred}, 置信度={final_conf:.4f}")
            else:
                print(f"  theta={theta:5.1f}: ❌ 失败")
        except Exception as e:
            print(f"  theta={theta:5.1f}: ❌ 错误 - {e}")
    
    # 3. SparseFool - 测试不同perturbation
    print("\n📍 SparseFool攻击 - 当前perturbation=0.8")
    print("-" * 80)
    try:
        success, adv, pixels = sparsefool_attack_simple(image, label, model, max_pixels=10)
        if success:
            l0 = compute_l0_norm(image.cpu(), adv.cpu())
            with torch.no_grad():
                final_pred = model(adv.unsqueeze(0)).argmax(dim=1).item()
                final_conf = torch.softmax(model(adv.unsqueeze(0)), dim=1)[0, final_pred].item()
            print(f"  ✅ 成功, L0={l0}, 最终预测={final_pred}, 置信度={final_conf:.4f}")
        else:
            print(f"  ❌ 失败")
    except Exception as e:
        print(f"  ❌ 错误 - {e}")
    
    # 4. 检查模型的鲁棒性
    print("\n" + "=" * 80)
    print("📊 模型鲁棒性分析")
    print("=" * 80)
    
    # 测试多个样本的置信度分布
    print("\n收集前50个正确分类样本的置信度...")
    confidences = []
    
    for idx in range(min(50, len(testset))):
        img, lbl = testset[idx]
        img = img.to(device)
        
        with torch.no_grad():
            out = model(img.unsqueeze(0))
            pred = out.argmax(dim=1).item()
            
            if pred == lbl:
                conf = torch.softmax(out, dim=1)[0, pred].item()
                confidences.append(conf)
    
    import numpy as np
    print(f"  样本数: {len(confidences)}")
    print(f"  平均置信度: {np.mean(confidences):.4f}")
    print(f"  中位数: {np.median(confidences):.4f}")
    print(f"  最小值: {np.min(confidences):.4f}")
    print(f"  最大值: {np.max(confidences):.4f}")
    
    print("\n" + "=" * 80)
    print("💡 结论和建议")
    print("=" * 80)
    print("""
1. 模型准确率提升到88.8%，确实更难攻击
2. 需要增加攻击强度：
   - One-Pixel: 增加maxiter到200-400
   - JSMA: 增加theta到10-50
   - SparseFool: 可能需要调整perturbation
   
3. 如果高置信度样本多，说明模型很"自信"，需要更强的攻击

4. 建议：
   - 使用找到的有效参数运行完整实验
   - 或者考虑使用84%模型进行初步论文实验
   - 88.8%模型可以作为"鲁棒性验证"部分
    """)


if __name__ == "__main__":
    debug_single_sample()

