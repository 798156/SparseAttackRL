"""
优化One-Pixel攻击 - 提升ASR
测试不同的max_iter参数
"""

import torch
import torchvision
import torchvision.transforms as transforms
from target_model import load_target_model
from one_pixel_attack import one_pixel_attack
from evaluation_metrics import compute_l0_norm
from tqdm import tqdm
import time
import json
import os

def test_onepixel_with_params(num_samples=100, max_iter=400):
    """
    测试One-Pixel with不同参数
    
    Args:
        num_samples: 测试样本数
        max_iter: 最大迭代次数
    """
    print("=" * 80)
    print(f"🔬 优化One-Pixel攻击 - max_iter={max_iter}")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载数据
    print("\n📦 加载数据...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    
    # 加载模型
    print("\n🔧 加载ResNet18模型...")
    model = load_target_model("resnet18")
    model = model.to(device)
    model.eval()
    
    # 收集正确分类的样本
    print(f"\n🔍 收集{num_samples}个正确分类的样本...")
    correct_samples = []
    
    for idx, (images, labels) in enumerate(testloader):
        if len(correct_samples) >= num_samples:
            break
        
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            pred = outputs.argmax(dim=1).item()
            
            if pred == labels.item():
                correct_samples.append((images[0], labels.item()))
    
    print(f"✅ 收集到 {len(correct_samples)} 个样本\n")
    
    # 运行One-Pixel攻击
    print(f"🚀 开始One-Pixel攻击测试 (max_iter={max_iter})...")
    print(f"预计时间: {len(correct_samples) * max_iter * 0.06 / 60:.1f} 分钟\n")
    
    successes = []
    times = []
    
    for idx, (image, label) in enumerate(tqdm(correct_samples, desc="攻击进度")):
        image = image.to(device)
        
        start = time.time()
        try:
            success, params = one_pixel_attack(image, label, model, max_iter=max_iter)
            elapsed = time.time() - start
            
            successes.append(success)
            times.append(elapsed)
        except Exception as e:
            print(f"\n样本 {idx} 失败: {e}")
            successes.append(False)
            times.append(0)
    
    # 统计结果
    import numpy as np
    asr = np.mean(successes) * 100
    avg_time = np.mean(times)
    
    print("\n" + "=" * 80)
    print("📊 One-Pixel优化结果")
    print("=" * 80)
    print(f"\nmax_iter: {max_iter}")
    print(f"ASR: {asr:.1f}% ({sum(successes)}/{len(successes)})")
    print(f"平均时间: {avg_time:.2f}秒")
    print(f"总时间: {sum(times)/60:.1f}分钟")
    
    # 保存结果
    result = {
        'max_iter': max_iter,
        'num_samples': len(correct_samples),
        'asr': float(asr),
        'avg_time': float(avg_time),
        'total_time': float(sum(times)),
        'successes': int(sum(successes))
    }
    
    output_dir = "results/onepixel_optimization"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/onepixel_maxiter_{max_iter}.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ 结果保存到: {output_dir}/onepixel_maxiter_{max_iter}.json")
    
    return result


def compare_max_iters():
    """对比不同max_iter的效果"""
    print("=" * 80)
    print("🔬 One-Pixel参数优化实验")
    print("=" * 80)
    print("\n我们将测试以下配置：")
    print("1. max_iter=200 (当前)")
    print("2. max_iter=400 (推荐)")
    print("3. max_iter=600 (可选)")
    print("\n每个配置测试30个样本（快速验证）")
    
    input("\n按Enter开始实验...")
    
    results = []
    
    # 测试不同的max_iter
    for max_iter in [200, 400, 600]:
        print("\n" + "=" * 80)
        print(f"📍 测试 max_iter={max_iter}")
        print("=" * 80)
        
        result = test_onepixel_with_params(num_samples=30, max_iter=max_iter)
        results.append(result)
        
        print(f"\n当前结果: ASR={result['asr']:.1f}%, 平均时间={result['avg_time']:.2f}秒")
    
    # 汇总对比
    print("\n" + "=" * 80)
    print("📊 参数对比总结")
    print("=" * 80)
    print(f"\n{'max_iter':<12} {'ASR (%)':<12} {'平均时间(s)':<15} {'推荐':<10}")
    print("-" * 60)
    
    for result in results:
        rec = ""
        if result['max_iter'] == 400:
            rec = "✅ 推荐"
        
        print(f"{result['max_iter']:<12} {result['asr']:<12.1f} {result['avg_time']:<15.2f} {rec:<10}")
    
    # 给出建议
    print("\n" + "=" * 80)
    print("💡 建议")
    print("=" * 80)
    
    best = max(results, key=lambda x: x['asr'])
    
    print(f"\n最佳配置: max_iter={best['max_iter']}")
    print(f"  ASR: {best['asr']:.1f}%")
    print(f"  100样本预计时间: {best['avg_time'] * 100 / 60:.1f}分钟")
    
    if best['asr'] >= 30:
        print(f"\n✅ 这个ASR（{best['asr']:.1f}%）已经足够用于论文！")
        print(f"建议使用 max_iter={best['max_iter']} 重新运行100样本实验")
    else:
        print(f"\n⚠️ ASR仍然偏低（{best['asr']:.1f}%）")
        print("建议考虑：")
        print("1. 继续增加max_iter到800")
        print("2. 或者在论文中讨论One-Pixel的局限性")


if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print("🎯 One-Pixel优化工具")
    print("=" * 80)
    print("\n选择运行模式：")
    print("1. 快速对比测试（30样本×3种参数）- 推荐先做这个")
    print("2. 完整测试（100样本，指定max_iter）")
    print()
    
    choice = input("请选择 (1 或 2): ").strip()
    
    if choice == "1":
        compare_max_iters()
    elif choice == "2":
        max_iter = int(input("请输入max_iter (推荐400-800): "))
        test_onepixel_with_params(num_samples=100, max_iter=max_iter)
    else:
        print("无效选择")




















