"""
One-Pixel攻击修复和增强测试
专门针对VGG16的0% ASR问题
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time

# 导入One-Pixel攻击
from one_pixel_attack import one_pixel_attack

def load_model():
    """加载VGG16模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载VGG16
    model = torchvision.models.vgg16(weights=None)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 10)
    
    # 加载权重
    model.load_state_dict(torch.load('cifar10_vgg16.pth', map_location=device))
    model.to(device)
    model.eval()
    
    return model, device

def load_test_data(num_samples=50):
    """加载测试数据"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    return testset

def test_onepixel_config(model, device, testset, config_name, max_iter, pop_size, num_samples=50):
    """测试特定配置的One-Pixel攻击"""
    print(f"\n{'='*80}")
    print(f"🧪 测试配置: {config_name}")
    print(f"{'='*80}")
    print(f"参数: maxiter={max_iter}, popsize={pop_size}")
    print(f"样本数: {num_samples}")
    
    # 选择正确分类的样本
    selected_samples = []
    for idx in range(len(testset)):
        if len(selected_samples) >= num_samples:
            break
        
        image, label = testset[idx]
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_batch)
            pred = output.argmax(dim=1).item()
        
        if pred == label:
            selected_samples.append((idx, image, label))
    
    print(f"✅ 选择了 {len(selected_samples)} 个正确分类的样本")
    
    # 测试攻击
    success_count = 0
    total_time = 0
    l0_values = []
    
    results = []
    
    for i, (idx, image, label) in enumerate(tqdm(selected_samples, desc="攻击进度")):
        start_time = time.time()
        
        # One-Pixel攻击 (注意参数顺序和返回值)
        success, adv_image, modified_info = one_pixel_attack(
            image=image,
            label=label,
            model=model,
            max_iter=max_iter,
            pop_size=pop_size
        )
        
        attack_time = time.time() - start_time
        
        # 验证攻击是否成功
        if success:
            # 计算L0
            diff = (adv_image - image).abs()
            modified_pixels = (diff.sum(dim=0) > 0).sum().item()
            l0_values.append(modified_pixels)
            success_count += 1
            
            results.append({
                'sample_id': idx,
                'success': True,
                'l0': modified_pixels,
                'time': attack_time,
                'modified_pixel': modified_info
            })
        else:
            results.append({
                'sample_id': idx,
                'success': False,
                'time': attack_time
            })
        
        total_time += attack_time
    
    # 统计结果
    asr = success_count / len(selected_samples) * 100
    avg_time = total_time / len(selected_samples)
    avg_l0 = np.mean(l0_values) if l0_values else 0
    
    print(f"\n📊 结果:")
    print(f"  ASR: {success_count}/{len(selected_samples)} = {asr:.1f}%")
    print(f"  平均L0: {avg_l0:.2f}")
    print(f"  平均时间: {avg_time:.2f}秒")
    
    return {
        'config': config_name,
        'max_iter': max_iter,
        'pop_size': pop_size,
        'asr': asr,
        'success_count': success_count,
        'total_samples': len(selected_samples),
        'avg_l0': avg_l0,
        'avg_time': avg_time,
        'detailed_results': results
    }

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔧 One-Pixel攻击修复 - VGG16专项测试")
    print("="*80)
    
    # 加载模型
    print("\n📦 加载VGG16模型...")
    model, device = load_model()
    print(f"✅ 模型加载完成，设备: {device}")
    
    # 加载数据
    print("\n📦 加载测试数据...")
    testset = load_test_data()
    print(f"✅ 数据加载完成")
    
    # 测试不同配置
    configs = [
        # 原始配置
        ("原始配置", 100, 400, 30),
        
        # 增加迭代次数
        ("增加迭代1", 200, 400, 30),
        ("增加迭代2", 300, 400, 30),
        
        # 增加种群大小
        ("增加种群1", 100, 800, 30),
        ("增加种群2", 100, 1200, 30),
        
        # 同时增加
        ("双倍配置", 200, 800, 30),
        ("三倍配置", 300, 1200, 30),
    ]
    
    all_results = []
    
    for config_name, max_iter, pop_size, num_samples in configs:
        result = test_onepixel_config(
            model=model,
            device=device,
            testset=testset,
            config_name=config_name,
            max_iter=max_iter,
            pop_size=pop_size,
            num_samples=num_samples
        )
        all_results.append(result)
        
        # 如果找到有效配置，可以提前停止
        if result['asr'] > 5.0:
            print(f"\n✅ 找到有效配置！ASR={result['asr']:.1f}%")
            print(f"继续测试剩余配置以找到最佳参数...")
    
    # 汇总结果
    print(f"\n{'='*80}")
    print("📊 所有配置汇总")
    print(f"{'='*80}\n")
    
    print(f"{'配置':<15} {'MaxIter':<10} {'PopSize':<10} {'ASR':<10} {'平均L0':<10} {'平均时间'}")
    print("-"*80)
    for r in all_results:
        print(f"{r['config']:<15} {r['max_iter']:<10} {r['pop_size']:<10} "
              f"{r['asr']:<10.1f} {r['avg_l0']:<10.2f} {r['avg_time']:.2f}s")
    
    # 找到最佳配置
    best_config = max(all_results, key=lambda x: x['asr'])
    print(f"\n{'='*80}")
    print(f"🏆 最佳配置: {best_config['config']}")
    print(f"{'='*80}")
    print(f"  MaxIter: {best_config['max_iter']}")
    print(f"  PopSize: {best_config['pop_size']}")
    print(f"  ASR: {best_config['asr']:.1f}%")
    print(f"  平均L0: {best_config['avg_l0']:.2f}")
    print(f"  平均时间: {best_config['avg_time']:.2f}秒")
    
    # 如果最佳配置ASR > 0，用它进行完整测试（100样本）
    if best_config['asr'] > 0:
        print(f"\n{'='*80}")
        print("🚀 使用最佳配置进行完整测试（100样本）")
        print(f"{'='*80}")
        
        final_result = test_onepixel_config(
            model=model,
            device=device,
            testset=testset,
            config_name="最终配置",
            max_iter=best_config['max_iter'],
            pop_size=best_config['pop_size'],
            num_samples=100
        )
        
        # 保存结果
        output_dir = Path('results/onepixel_fix')
        output_dir.mkdir(exist_ok=True, parents=True)
        
        with open(output_dir / 'vgg16_onepixel_optimized.json', 'w') as f:
            json.dump(final_result, f, indent=2)
        
        print(f"\n✅ 最终结果已保存到: {output_dir / 'vgg16_onepixel_optimized.json'}")
    else:
        print(f"\n{'='*80}")
        print("⚠️  所有配置的ASR都是0%")
        print(f"{'='*80}")
        print("\n这可能意味着:")
        print("  1. VGG16对One-Pixel攻击确实高度鲁棒")
        print("  2. 这是一个重要的研究发现！")
        print("  3. 论文中需要详细讨论这个现象")
    
    # 保存所有配置的结果
    output_dir = Path('results/onepixel_fix')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / 'all_configs_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ 所有配置对比已保存到: {output_dir / 'all_configs_comparison.json'}")
    
    print(f"\n{'='*80}")
    print("🎉 One-Pixel修复测试完成！")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

