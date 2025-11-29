"""
完整的5种稀疏攻击方法对比测试
3个模型 × 5种攻击方法 × 30样本
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
from pgd_l0_attack import pgd_l0_attack
from pixel_gradient_attack import pixel_gradient_attack

# ============= 统一参数配置 =============
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
    'PGD-L0': {
        'max_pixels': 10,
        'step_size': 0.1,
        'num_steps': 20
    },
    'PixelGrad': {
        'max_pixels': 10,
        'alpha': 0.2,
        'beta': 0.9
    },
    'test_samples': 30,
    'random_seed': 42
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
    
    # 打印参数
    if method_name == 'JSMA':
        print(f"参数: max_pixels={params['max_pixels']}, theta={params['theta']}")
    elif method_name == 'SparseFool':
        print(f"参数: max_iterations={params['max_iterations']}, lambda_={params['lambda_']}")
    elif method_name == 'Greedy':
        print(f"参数: max_pixels={params['max_pixels']}, step_size={params['step_size']}")
    elif method_name == 'PGD-L0':
        print(f"参数: max_pixels={params['max_pixels']}, step_size={params['step_size']}, num_steps={params['num_steps']}")
    elif method_name == 'PixelGrad':
        print(f"参数: max_pixels={params['max_pixels']}, alpha={params['alpha']}, beta={params['beta']}")
    
    # 选择样本
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
            if method_name == 'JSMA':
                success, adv_image, modified_pixels = jsma_attack(
                    image=image, label=label, model=model,
                    max_pixels=params['max_pixels'], theta=params['theta']
                )
            elif method_name == 'SparseFool':
                success, adv_image, modified_pixels = sparsefool_attack(
                    image=image, label=label, model=model,
                    max_iterations=params['max_iterations'], lambda_=params['lambda_']
                )
            elif method_name == 'Greedy':
                success, adv_image, modified_pixels = greedy_attack(
                    image=image, label=label, model=model,
                    max_pixels=params['max_pixels'], step_size=params['step_size']
                )
            elif method_name == 'PGD-L0':
                success, adv_image, modified_pixels = pgd_l0_attack(
                    image=image, label=label, model=model,
                    max_pixels=params['max_pixels'], 
                    step_size=params['step_size'],
                    num_steps=params['num_steps']
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
    print("🔬 完整5种攻击方法对比测试")
    print("="*80)
    print("\n💡 攻击方法:")
    print("  1. JSMA - 雅可比显著性图攻击")
    print("  2. SparseFool - 几何边界优化")
    print("  3. Greedy - 贪心梯度攻击")
    print("  4. PGD-L0 - 投影梯度下降L0约束")
    print("  5. PixelGrad - 像素梯度动量攻击")
    print(f"\n📊 实验规模: 3模型 × 5方法 × 30样本 = 450个测试")
    print(f"⏰ 预计时间: 10-15分钟\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}\n")
    
    # 加载数据
    print("📦 加载CIFAR-10数据...")
    testset = load_cifar10_data()
    print(f"✅ 完成\n")
    
    # 测试矩阵
    models = ['ResNet18', 'VGG16', 'MobileNetV2']
    methods = ['JSMA', 'SparseFool', 'Greedy', 'PGD-L0', 'PixelGrad']
    
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
            
            # 保存中间结果
            output_dir = Path('results/5methods_baseline')
            output_dir.mkdir(exist_ok=True, parents=True)
            
            with open(output_dir / f'{model_name.lower()}_{method_name.lower()}.json', 'w') as f:
                json.dump(result, f, indent=2)
    
    total_time = time.time() - start_time_total
    
    # 汇总
    print(f"\n{'='*80}")
    print("📊 完整实验结果汇总")
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
    
    # 按方法对比
    print(f"\n{'='*80}")
    print("📊 按攻击方法对比（平均值）")
    print(f"{'='*80}\n")
    
    for method_name in methods:
        method_results = [r for r in all_results if r['method'] == method_name]
        avg_asr = np.mean([r['asr'] for r in method_results])
        avg_l0 = np.mean([r['avg_l0'] for r in method_results if r['avg_l0'] > 0])
        avg_time = np.mean([r['avg_time'] for r in method_results])
        
        print(f"【{method_name}】")
        print(f"  平均ASR: {avg_asr:.1f}%")
        print(f"  平均L0: {avg_l0:.2f}")
        print(f"  平均时间: {avg_time:.3f}秒")
        print()
    
    # 保存汇总
    summary = {
        'config': CONFIG,
        'total_time_minutes': float(total_time/60),
        'device': str(device),
        'note': 'Complete comparison with 5 sparse attack methods',
        'results': all_results
    }
    
    with open(output_dir / 'complete_5methods_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 所有结果已保存到: {output_dir}")
    
    print(f"\n{'='*80}")
    print("🎉 完整测试完成！")
    print(f"{'='*80}")
    print("\n✅ 获得的数据:")
    print("  - 3个模型（ResNet18, VGG16, MobileNetV2）")
    print("  - 5种方法（JSMA, SparseFool, Greedy, PGD-L0, PixelGrad）")
    print("  - 每个组合30个样本")
    print("  - 总计450个测试")
    print("  - 统一的参数配置")
    print("\n💡 这是更全面的对比研究数据！\n")

if __name__ == "__main__":
    main()
















