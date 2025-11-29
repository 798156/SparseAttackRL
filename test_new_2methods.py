"""
只测试2个新增的攻击方法
- RandomSparse
- PixelGrad

3个模型 × 2个新方法 × 100样本 = 600个测试
结果可以和之前的3个方法合并
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

from random_sparse_attack import random_sparse_attack_smart
from pixel_gradient_attack import pixel_gradient_attack

# ============= 参数配置（与之前保持一致）=============
CONFIG = {
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
    'test_samples': 100,  # 从30增加到100，提高统计可靠性
    'random_seed': 42  # 与之前相同，确保选择相同的样本
}

def load_cifar10_data():
    """加载CIFAR-10测试数据"""
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    return testset

def load_model(model_name, device):
    """加载指定模型"""
    if model_name == 'ResNet18':
        model = torchvision.models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        model.load_state_dict(torch.load('cifar10_resnet18.pth', map_location=device, weights_only=False))
    
    elif model_name == 'VGG16':
        model = torchvision.models.vgg16(weights=None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 10)
        model.load_state_dict(torch.load('cifar10_vgg16.pth', map_location=device, weights_only=False))
    
    elif model_name == 'MobileNetV2':
        model = torchvision.models.mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 10)
        model.load_state_dict(torch.load('cifar10_mobilenetv2.pth', map_location=device, weights_only=False))
    
    model.to(device)
    model.eval()
    return model

def calculate_metrics(original, adversarial):
    """计算评估指标"""
    orig_cpu = original.detach().cpu()
    adv_cpu = adversarial.detach().cpu()
    
    # L0范数
    diff = (adv_cpu - orig_cpu).abs()
    l0 = (diff.sum(dim=0) > 1e-5).sum().item()
    
    # L2范数
    l2 = torch.norm(diff).item()
    
    # SSIM
    orig_np = orig_cpu.numpy().transpose(1, 2, 0)
    adv_np = adv_cpu.numpy().transpose(1, 2, 0)
    
    orig_np = (orig_np - orig_np.min()) / (orig_np.max() - orig_np.min() + 1e-8)
    adv_np = (adv_np - adv_np.min()) / (adv_np.max() - adv_np.min() + 1e-8)
    
    ssim_value = ssim_func(orig_np, adv_np, channel_axis=2, data_range=1.0)
    
    return l0, l2, ssim_value

def test_single_method_model(method_name, model_name, model, testset, device, num_samples=30):
    """测试单个方法在单个模型上的表现"""
    print(f"\n{'='*80}")
    print(f"🎯 {model_name} + {method_name}")
    print(f"{'='*80}")
    
    # 获取参数
    params = CONFIG[method_name]
    
    if method_name == 'RandomSparse':
        print(f"参数: max_pixels={params['max_pixels']}, perturbation_size={params['perturbation_size']}, max_attempts={params['max_attempts']}")
    elif method_name == 'PixelGrad':
        print(f"参数: max_pixels={params['max_pixels']}, alpha={params['alpha']}, beta={params['beta']}")
    
    # 选择样本（使用相同的随机种子，确保和之前测试相同的样本）
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    
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
    
    print(f"✅ 测试 {len(selected_samples)} 个样本\n")
    
    # 执行攻击
    success_count = 0
    results = {
        'l0': [],
        'l2': [],
        'ssim': [],
        'time': [],
        'details': []
    }
    
    for i, (idx, image, label) in enumerate(tqdm(selected_samples, desc=method_name)):
        start_time = time.time()
        
        try:
            if method_name == 'RandomSparse':
                success, adv_image, modified_pixels = random_sparse_attack_smart(
                    image=image, label=label, model=model,
                    max_pixels=params['max_pixels'],
                    perturbation_size=params['perturbation_size'],
                    max_attempts=params['max_attempts']
                )
            elif method_name == 'PixelGrad':
                success, adv_image, modified_pixels = pixel_gradient_attack(
                    image=image, label=label, model=model,
                    max_pixels=params['max_pixels'],
                    alpha=params['alpha'],
                    beta=params['beta']
                )
            
            attack_time = time.time() - start_time
            
            if success:
                l0, l2, ssim_val = calculate_metrics(image, adv_image)
                
                results['l0'].append(l0)
                results['l2'].append(l2)
                results['ssim'].append(ssim_val)
                results['time'].append(attack_time)
                
                success_count += 1
                
                results['details'].append({
                    'sample_id': int(idx),
                    'success': True,
                    'l0': float(l0),
                    'l2': float(l2),
                    'ssim': float(ssim_val),
                    'time': float(attack_time)
                })
            else:
                results['time'].append(attack_time)
                results['details'].append({
                    'sample_id': int(idx),
                    'success': False,
                    'time': float(attack_time)
                })
        
        except Exception as e:
            print(f"\n⚠️  样本{i}出错: {str(e)}")
            results['details'].append({
                'sample_id': int(idx),
                'success': False,
                'error': str(e)
            })
    
    # 统计
    asr = success_count / len(selected_samples) * 100
    avg_l0 = np.mean(results['l0']) if results['l0'] else 0
    avg_l2 = np.mean(results['l2']) if results['l2'] else 0
    avg_ssim = np.mean(results['ssim']) if results['ssim'] else 0
    avg_time = np.mean(results['time']) if results['time'] else 0
    
    print(f"\n📊 结果:")
    print(f"  ASR: {success_count}/{len(selected_samples)} = {asr:.1f}%")
    print(f"  平均L0: {avg_l0:.2f}")
    print(f"  平均L2: {avg_l2:.4f}")
    print(f"  平均SSIM: {avg_ssim:.4f}")
    print(f"  平均时间: {avg_time:.3f}秒")
    
    return {
        'model': model_name,
        'method': method_name,
        'parameters': params,
        'asr': float(asr),
        'success_count': success_count,
        'total_samples': len(selected_samples),
        'avg_l0': float(avg_l0),
        'avg_l2': float(avg_l2),
        'avg_ssim': float(avg_ssim),
        'avg_time': float(avg_time),
        'std_l0': float(np.std(results['l0'])) if results['l0'] else 0,
        'std_l2': float(np.std(results['l2'])) if results['l2'] else 0,
        'detailed_results': results['details']
    }

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔬 测试2个新增攻击方法")
    print("="*80)
    print("\n💡 新增方法:")
    print("  1. RandomSparse - 随机稀疏攻击（baseline）")
    print("  2. PixelGrad - 像素梯度动量攻击")
    print(f"\n📊 实验规模: 3模型 × 2方法 × 30样本 = 180个测试")
    print(f"⏰ 预计时间: 3-5分钟\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}\n")
    
    # 加载数据
    print("📦 加载CIFAR-10数据...")
    testset = load_cifar10_data()
    print(f"✅ 完成\n")
    
    # 测试矩阵
    models = ['ResNet18', 'VGG16', 'MobileNetV2']
    methods = ['RandomSparse', 'PixelGrad']
    
    all_results = []
    start_time_total = time.time()
    
    for model_name in models:
        print(f"\n{'='*80}")
        print(f"📦 加载模型: {model_name}")
        print(f"{'='*80}")
        
        model = load_model(model_name, device)
        
        for method_name in methods:
            result = test_single_method_model(
                method_name=method_name,
                model_name=model_name,
                model=model,
                testset=testset,
                device=device,
                num_samples=CONFIG['test_samples']
            )
            all_results.append(result)
            
            # 保存结果（与之前的结果放在一起）
            output_dir = Path('results/complete_baseline')
            output_dir.mkdir(exist_ok=True, parents=True)
            
            method_filename = method_name.lower().replace('-', '')  # pgd-l0 -> pgdl0
            with open(output_dir / f'{model_name.lower()}_{method_filename}.json', 'w') as f:
                json.dump(result, f, indent=2)
    
    total_time = time.time() - start_time_total
    
    # 汇总
    print(f"\n{'='*80}")
    print("📊 新方法测试结果汇总")
    print(f"{'='*80}")
    print(f"总耗时: {total_time/60:.1f}分钟\n")
    
    for model_name in models:
        print(f"\n【{model_name}】")
        print(f"{'方法':<15} {'ASR':<8} {'平均L0':<10} {'平均L2':<10} {'平均SSIM':<12} {'时间'}")
        print("-"*75)
        for r in all_results:
            if r['model'] == model_name:
                print(f"{r['method']:<15} {r['asr']:<8.1f} {r['avg_l0']:<10.2f} "
                      f"{r['avg_l2']:<10.4f} {r['avg_ssim']:<12.4f} {r['avg_time']:.3f}s")
    
    # 保存新方法汇总
    summary = {
        'config': CONFIG,
        'total_time_minutes': float(total_time/60),
        'device': str(device),
        'note': 'New 2 methods: PGD-L0 and PixelGrad',
        'results': all_results
    }
    
    with open(output_dir / 'new_2methods_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 结果已保存到: {output_dir}")
    
    print(f"\n{'='*80}")
    print("🎉 新方法测试完成！")
    print(f"{'='*80}")
    print("\n✅ 测试数据:")
    print("  - 2个新方法（RandomSparse, PixelGrad）")
    print("  - 3个模型（ResNet18, VGG16, MobileNetV2）")
    print("  - 每个组合30个样本")
    print("  - 总计180个新测试")
    print("\n💡 可以与之前的3个方法（JSMA, SparseFool, Greedy）合并分析")
    print("   → 总共5个方法的完整对比数据")
    print("\n🎯 RandomSparse作为baseline，证明智能方法的优越性！\n")

if __name__ == "__main__":
    main()

