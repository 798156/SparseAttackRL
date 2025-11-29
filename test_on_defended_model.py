"""
在对抗训练的防御模型上测试5种稀疏攻击方法

目的：
1. 证明方法在防御模型上也有效
2. 对比标准模型 vs 防御模型的ASR差异
3. 验证方法的相对性能是否保持

测试：1个防御模型 × 5种方法 × 100样本 = 500个测试
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
from skimage.metrics import structural_similarity as ssim_func

from jsma_attack import jsma_attack
from sparsefool_attack import sparsefool_attack
from greedy_attack import greedy_attack
from random_sparse_attack import random_sparse_attack_smart
from pixel_gradient_attack import pixel_gradient_attack

# ============= 配置 =============
CONFIG = {
    'JSMA': {
        'max_pixels': 10,
        'theta': 1.0
    },
    'SparseFool': {
        'max_iterations': 20,
        'lambda_': 3.0
    },
    'Greedy': {
        'max_pixels': 10,
        'step_size': 0.2
    },
    'RandomSparse': {
        'max_pixels': 10,
        'perturbation_size': 0.2,
        'max_attempts': 50
    },
    'PixelGrad': {
        'max_pixels': 10,
        'alpha': 0.2,
        'beta': 0.9
    },
    'test_samples': 100,
    'random_seed': 42,
    
    # 防御模型选择
    'defense_type': 'pgd',  # 'trades' 或 'pgd' 或 'custom' - 切换到更小的Wong2020Fast模型（~150MB）
    'defense_model_path': 'cifar10_resnet18_defended.pth'  # 如果使用custom
}

def load_cifar10_data():
    """加载CIFAR-10测试数据"""
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    return testset

def load_defended_model(defense_type, device):
    """
    加载防御模型
    
    方案1：使用预训练的对抗训练模型（推荐）
    方案2：使用自己训练的模型
    """
    print(f"📦 加载防御模型: {defense_type}")
    
    if defense_type in ['trades', 'pgd']:
        # 方案1：尝试使用RobustBench（如果安装了）
        try:
            from robustbench.utils import load_model as load_robust_model
            print("  → 使用RobustBench预训练模型")
            
            if defense_type == 'trades':
                model_name = 'Rice2020Overfitting'  # TRADES训练的ResNet18
            else:  # pgd
                model_name = 'Wong2020Fast'  # PGD对抗训练
            
            model = load_robust_model(
                model_name=model_name,
                dataset='cifar10',
                threat_model='Linf'
            )
            model = model.to(device)
            model.eval()
            print(f"  ✅ 成功加载 {model_name}")
            return model
            
        except ImportError:
            print("  ⚠️ RobustBench未安装，使用自训练模型")
            return load_custom_defended_model(device)
    
    elif defense_type == 'custom':
        # 方案2：使用自己训练的防御模型
        return load_custom_defended_model(device)
    
    else:
        raise ValueError(f"未知的防御类型: {defense_type}")

def load_custom_defended_model(device):
    """加载自定义训练的防御模型"""
    model_path = CONFIG['defense_model_path']
    
    if not Path(model_path).exists():
        print(f"\n{'='*80}")
        print("❌ 错误：防御模型不存在！")
        print(f"{'='*80}")
        print(f"模型路径: {model_path}")
        print("\n请选择以下方案之一：")
        print("\n方案1：使用简单防御模型（推荐，快速）")
        print("  → 运行: python create_simple_defended_model.py")
        print("  → 这会创建一个基本的防御模型（5-10分钟）")
        print("\n方案2：完整对抗训练（耗时但更好）")
        print("  → 运行: python train_adversarial_resnet18.py")
        print("  → 需要4-6小时")
        print("\n方案3：安装RobustBench使用预训练模型（最简单）")
        print("  → pip install robustbench")
        print("  → 修改CONFIG['defense_type'] = 'trades'")
        print(f"{'='*80}")
        raise FileNotFoundError(f"防御模型不存在: {model_path}")
    
    # 加载模型
    model = torchvision.models.resnet18(num_classes=10)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print(f"  ✅ 成功加载自定义防御模型")
    return model

def select_test_samples(testset, num_samples, model, device, seed=42):
    """选择模型正确分类的样本"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"📋 选择测试样本...")
    correct_indices = []
    
    for idx in range(len(testset)):
        image, label = testset[idx]
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_batch)
            pred = output.argmax(dim=1).item()
        
        if pred == label:
            correct_indices.append(idx)
        
        if len(correct_indices) >= num_samples * 2:
            break
    
    selected = np.random.choice(correct_indices, size=num_samples, replace=False)
    print(f"  ✅ 从 {len(correct_indices)} 个正确样本中选择了 {num_samples} 个")
    return selected.tolist()

def calculate_ssim(img1, img2):
    """计算SSIM"""
    img1_np = img1.cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.cpu().numpy().transpose(1, 2, 0)
    
    ssim_value = ssim_func(
        img1_np, img2_np,
        multichannel=True,
        data_range=1.0,
        channel_axis=2
    )
    return ssim_value

def test_attack_method(method_name, attack_func, params, model, testset, 
                       sample_indices, device):
    """测试单个攻击方法"""
    results = {
        'method': method_name,
        'params': params,
        'samples': []
    }
    
    print(f"\n{'='*80}")
    print(f"🎯 测试: {method_name}")
    print(f"{'='*80}")
    print(f"参数: {params}")
    print(f"✅ 测试 {len(sample_indices)} 个样本")
    
    for idx in tqdm(sample_indices, desc=method_name):
        image, label = testset[idx]
        image = image.to(device)
        
        start_time = time.time()
        
        try:
            success, adv_image, modified_pixels = attack_func(
                image, label, model, **params
            )
            elapsed = time.time() - start_time
            
            if success:
                # 计算指标
                l0_norm = len(modified_pixels)
                l2_norm = torch.norm(adv_image - image, p=2).item()
                ssim_value = calculate_ssim(image, adv_image)
                
                results['samples'].append({
                    'idx': int(idx),
                    'success': True,
                    'l0': l0_norm,
                    'l2': float(l2_norm),
                    'ssim': float(ssim_value),
                    'time': elapsed,
                    'modified_pixels': len(modified_pixels)
                })
            else:
                results['samples'].append({
                    'idx': int(idx),
                    'success': False,
                    'time': elapsed
                })
                
        except Exception as e:
            print(f"⚠️ 样本{idx}出错: {e}")
            results['samples'].append({
                'idx': int(idx),
                'success': False,
                'error': str(e)
            })
    
    # 统计
    successes = [s for s in results['samples'] if s.get('success', False)]
    asr = len(successes) / len(results['samples']) * 100
    
    if successes:
        avg_l0 = np.mean([s['l0'] for s in successes])
        avg_l2 = np.mean([s['l2'] for s in successes])
        avg_ssim = np.mean([s['ssim'] for s in successes])
        avg_time = np.mean([s['time'] for s in results['samples']])
    else:
        avg_l0 = avg_l2 = avg_ssim = avg_time = 0
    
    results['summary'] = {
        'asr': asr,
        'avg_l0': avg_l0,
        'avg_l2': avg_l2,
        'avg_ssim': avg_ssim,
        'avg_time': avg_time,
        'total_samples': len(results['samples'])
    }
    
    print(f"\n📊 结果:")
    print(f"  ASR: {len(successes)}/{len(results['samples'])} = {asr:.1f}%")
    if successes:
        print(f"  平均L0: {avg_l0:.2f}")
        print(f"  平均L2: {avg_l2:.4f}")
        print(f"  平均SSIM: {avg_ssim:.4f}")
    print(f"  平均时间: {avg_time:.3f}秒")
    
    return results

def main():
    """主流程"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🛡️  防御模型测试 - 5种攻击方法                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}")
    
    # 加载数据
    testset = load_cifar10_data()
    
    # 加载防御模型
    try:
        model = load_defended_model(CONFIG['defense_type'], device)
    except Exception as e:
        print(f"\n❌ 加载防御模型失败: {e}")
        print("\n💡 建议：运行快速创建脚本")
        print("  → python create_simple_defended_model.py")
        return
    
    # 选择测试样本
    sample_indices = select_test_samples(
        testset, CONFIG['test_samples'], model, device, CONFIG['random_seed']
    )
    
    # 准备攻击方法
    attack_methods = [
        ('JSMA', jsma_attack, CONFIG['JSMA']),
        ('SparseFool', sparsefool_attack, CONFIG['SparseFool']),
        ('Greedy', greedy_attack, CONFIG['Greedy']),
        ('RandomSparse', random_sparse_attack_smart, CONFIG['RandomSparse']),
        ('PixelGrad', pixel_gradient_attack, CONFIG['PixelGrad'])
    ]
    
    # 测试所有方法
    all_results = {}
    start_time = time.time()
    
    for method_name, attack_func, params in attack_methods:
        results = test_attack_method(
            method_name, attack_func, params,
            model, testset, sample_indices, device
        )
        all_results[method_name] = results
    
    total_time = time.time() - start_time
    
    # 保存结果
    output_dir = Path('results/defended_model')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for method_name, results in all_results.items():
        output_file = output_dir / f'defended_{method_name.lower()}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # 打印总结
    print(f"\n{'='*80}")
    print("📊 防御模型测试总结")
    print(f"{'='*80}")
    print(f"总耗时: {total_time/60:.1f}分钟\n")
    
    print(f"{'方法':<15} {'ASR':<10} {'平均L0':<10} {'平均L2':<12} {'平均SSIM':<10} {'时间(s)':<10}")
    print("-" * 80)
    
    for method_name, results in all_results.items():
        summary = results['summary']
        print(f"{method_name:<15} "
              f"{summary['asr']:<10.1f} "
              f"{summary['avg_l0']:<10.2f} "
              f"{summary['avg_l2']:<12.4f} "
              f"{summary['avg_ssim']:<10.4f} "
              f"{summary['avg_time']:<10.3f}")
    
    print(f"\n💾 结果已保存到: {output_dir}")
    print(f"\n{'='*80}")
    print("🎉 防御模型测试完成！")
    print(f"{'='*80}")
    print("\n📈 下一步：")
    print("  1. 运行对比分析: python compare_standard_vs_defended.py")
    print("  2. 生成论文图表")
    print("  3. 继续Week 1 Day 5数据整理")

if __name__ == "__main__":
    main()

