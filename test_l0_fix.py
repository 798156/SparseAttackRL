# test_l0_fix.py
"""测试L0范数计算和攻击方法的实际像素修改"""

import torch
import numpy as np
from target_model import load_target_model
from jsma_attack import jsma_attack
from one_pixel_attack import one_pixel_attack
from evaluation_metrics import compute_l0_norm, compute_all_metrics
import torchvision
import torchvision.transforms as transforms

def test_l0_calculation():
    """测试L0计算是否正确"""
    print("=" * 60)
    print("🧪 测试L0范数计算")
    print("=" * 60)
    
    # 创建测试图像
    original = torch.randn(3, 32, 32)
    
    # 测试1: 修改1个像素
    adv1 = original.clone()
    adv1[:, 10, 10] += 1.0
    l0_1 = compute_l0_norm(original, adv1)
    print(f"\n测试1 - 修改1个像素:")
    print(f"  实际修改: 1像素")
    print(f"  计算L0: {l0_1}")
    print(f"  ✅ 正确" if l0_1 == 1 else f"  ❌ 错误")
    
    # 测试2: 修改5个像素
    adv2 = original.clone()
    for i in range(5):
        adv2[:, i, i] += 1.0
    l0_2 = compute_l0_norm(original, adv2)
    print(f"\n测试2 - 修改5个像素:")
    print(f"  实际修改: 5像素")
    print(f"  计算L0: {l0_2}")
    print(f"  ✅ 正确" if l0_2 == 5 else f"  ❌ 错误")
    
    # 测试3: 未修改
    adv3 = original.clone()
    l0_3 = compute_l0_norm(original, adv3)
    print(f"\n测试3 - 未修改:")
    print(f"  实际修改: 0像素")
    print(f"  计算L0: {l0_3}")
    print(f"  ✅ 正确" if l0_3 == 0 else f"  ❌ 错误")
    

def test_attack_methods():
    """测试攻击方法是否真的修改了像素"""
    print("\n" + "=" * 60)
    print("🧪 测试攻击方法的实际像素修改")
    print("=" * 60)
    
    # 加载模型和数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_target_model("resnet18")
    model = model.to(device)
    model.eval()
    
    # 加载测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # 测试几个样本
    for idx in range(3):
        print(f"\n{'='*60}")
        print(f"样本 #{idx}")
        print(f"{'='*60}")
        
        image, label = testset[idx]
        
        # 检查原始预测
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
            print(f"  原始标签: {label}, 模型预测: {pred}")
            
            if pred != label:
                print(f"  ⚠️ 模型预测错误，跳过此样本")
                continue
        
        # 测试JSMA
        print(f"\n  🔍 测试 JSMA:")
        success_jsma, adv_jsma, modified_jsma = jsma_attack(
            image.to(device), label, model, max_pixels=5
        )
        
        # 检查图像是否真的被修改
        diff_jsma = torch.abs(image.cpu() - adv_jsma.cpu())
        max_diff = diff_jsma.max().item()
        sum_diff = diff_jsma.sum().item()
        
        l0_jsma = compute_l0_norm(image.cpu(), adv_jsma.cpu())
        
        print(f"    成功: {success_jsma}")
        print(f"    修改像素列表长度: {len(modified_jsma)}")
        print(f"    图像最大差异: {max_diff:.6f}")
        print(f"    图像总差异: {sum_diff:.6f}")
        print(f"    计算的L0: {l0_jsma}")
        print(f"    {'✅ 图像被修改' if max_diff > 1e-6 else '❌ 图像未被修改'}")
        
        # 如果成功了，检查新的预测
        if success_jsma:
            with torch.no_grad():
                output_adv = model(adv_jsma.unsqueeze(0).to(device))
                pred_adv = output_adv.argmax(dim=1).item()
                print(f"    对抗样本预测: {pred_adv}")
                print(f"    {'✅ 攻击成功' if pred_adv != label else '❌ 攻击失败'}")
        
        # 测试One-Pixel
        print(f"\n  🔍 测试 One-Pixel:")
        success_op, adv_op, modified_op = one_pixel_attack(
            image.to(device), label, model, max_iter=50, pixels=1
        )
        
        diff_op = torch.abs(image.cpu() - adv_op.cpu())
        max_diff_op = diff_op.max().item()
        sum_diff_op = diff_op.sum().item()
        
        l0_op = compute_l0_norm(image.cpu(), adv_op.cpu())
        
        print(f"    成功: {success_op}")
        print(f"    修改像素列表长度: {len(modified_op)}")
        print(f"    图像最大差异: {max_diff_op:.6f}")
        print(f"    图像总差异: {sum_diff_op:.6f}")
        print(f"    计算的L0: {l0_op}")
        print(f"    {'✅ 图像被修改' if max_diff_op > 1e-6 else '❌ 图像未被修改'}")
        
        if success_op:
            with torch.no_grad():
                output_adv_op = model(adv_op.unsqueeze(0).to(device))
                pred_adv_op = output_adv_op.argmax(dim=1).item()
                print(f"    对抗样本预测: {pred_adv_op}")
                print(f"    {'✅ 攻击成功' if pred_adv_op != label else '❌ 攻击失败'}")


if __name__ == '__main__':
    # 测试L0计算
    test_l0_calculation()
    
    # 测试攻击方法
    test_attack_methods()
    
    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)
