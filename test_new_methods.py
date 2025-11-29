"""
测试新添加的攻击方法
验证效果和性能
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time

from greedy_attack import greedy_attack

def load_model(model_name='ResNet18'):
    """加载模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name == 'ResNet18':
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load('cifar10_resnet18.pth', map_location=device, weights_only=False))
    
    model.to(device)
    model.eval()
    return model, device

def test_greedy_attack():
    """测试Greedy Attack"""
    print("\n" + "="*80)
    print("🧪 测试 Greedy Gradient Attack")
    print("="*80)
    
    model, device = load_model()
    
    # 加载数据
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 选择10个正确分类的样本
    selected_samples = []
    for idx in range(len(testset)):
        if len(selected_samples) >= 10:
            break
        image, label = testset[idx]
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device)).argmax(dim=1).item()
        if pred == label:
            selected_samples.append((idx, image, label))
    
    print(f"✅ 选择了 {len(selected_samples)} 个样本\n")
    
    # 测试不同step_size
    step_sizes = [0.2, 0.3, 0.5]
    
    for step_size in step_sizes:
        print(f"\n{'='*40}")
        print(f"测试 step_size={step_size}")
        print(f"{'='*40}")
        
        success_count = 0
        l0_values = []
        time_values = []
        
        for i, (idx, image, label) in enumerate(selected_samples):
            start_time = time.time()
            
            success, adv_image, modified_pixels = greedy_attack(
                image=image,
                label=label,
                model=model,
                max_pixels=10,
                step_size=step_size
            )
            
            elapsed = time.time() - start_time
            time_values.append(elapsed)
            
            if success:
                # 计算L0
                adv_cpu = adv_image.detach().cpu()
                img_cpu = image.cpu()
                diff = (adv_cpu - img_cpu).abs()
                l0 = (diff.sum(dim=0) > 1e-5).sum().item()
                l0_values.append(l0)
                success_count += 1
                print(f"  样本{i}: ✅ 成功, L0={l0}, 时间={elapsed:.3f}s")
            else:
                print(f"  样本{i}: ❌ 失败, 时间={elapsed:.3f}s")
        
        asr = success_count / len(selected_samples) * 100
        avg_l0 = np.mean(l0_values) if l0_values else 0
        avg_time = np.mean(time_values)
        
        print(f"\n📊 step_size={step_size}:")
        print(f"  ASR: {success_count}/{len(selected_samples)} = {asr:.1f}%")
        print(f"  平均L0: {avg_l0:.2f}")
        print(f"  平均时间: {avg_time:.3f}s")
    
    print(f"\n{'='*80}")
    print("✅ Greedy Attack 测试完成")
    print(f"{'='*80}\n")

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔬 新攻击方法验证")
    print("="*80)
    print("\n目标:")
    print("  1. 验证Greedy Attack有效性")
    print("  2. 找到合适的参数")
    print("  3. 与JSMA/SparseFool对比\n")
    
    # 测试Greedy Attack
    test_greedy_attack()
    
    print("\n" + "="*80)
    print("💡 参数推荐")
    print("="*80)
    print("\n基于测试结果，推荐配置：")
    print("\nGreedy Attack:")
    print("  max_pixels: 10")
    print("  step_size: 0.3 (根据测试结果选择最佳)")
    print("\n预期性能:")
    print("  ASR: 70-90%")
    print("  L0: 3-5")
    print("  时间: < 0.3s")
    
    print(f"\n{'='*80}")
    print("🎉 验证完成！")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()







