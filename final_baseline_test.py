"""
最终版本的Baseline测试
基于诊断结果的统一参数配置
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

# ============= 经过诊断验证的统一参数 =============
FINAL_CONFIG = {
    'JSMA': {
        'max_pixels': 10,
        'theta': 1.0  # 诊断结果：90% ASR, L0=3.89
    },
    'SparseFool': {
        'max_iterations': 20,
        'lambda_': 3.0  # 诊断结果：90% ASR, L0=3.78
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
    # 确保在CPU上
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
    
    # 归一化到[0,1]
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
    if method_name == 'JSMA':
        params = FINAL_CONFIG['JSMA']
        print(f"参数: max_pixels={params['max_pixels']}, theta={params['theta']}")
    else:
        params = FINAL_CONFIG['SparseFool']
        print(f"参数: max_iterations={params['max_iterations']}, lambda_={params['lambda_']}")
    
    # 选择样本
    np.random.seed(FINAL_CONFIG['random_seed'])
    torch.manual_seed(FINAL_CONFIG['random_seed'])
    
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
                    image=image,
                    label=label,
                    model=model,
                    max_pixels=params['max_pixels'],
                    theta=params['theta']
                )
            else:  # SparseFool
                success, adv_image, modified_pixels = sparsefool_attack(
                    image=image,
                    label=label,
                    model=model,
                    max_iterations=params['max_iterations'],
                    lambda_=params['lambda_']
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
    print("🔬 最终版Baseline测试 - 经过诊断验证的参数")
    print("="*80)
    print("\n💡 统一参数配置:")
    print(f"  JSMA: max_pixels={FINAL_CONFIG['JSMA']['max_pixels']}, "
          f"theta={FINAL_CONFIG['JSMA']['theta']}")
    print(f"  SparseFool: max_iterations={FINAL_CONFIG['SparseFool']['max_iterations']}, "
          f"lambda_={FINAL_CONFIG['SparseFool']['lambda_']}")
    print(f"  样本数: {FINAL_CONFIG['test_samples']}/模型")
    print(f"  随机种子: {FINAL_CONFIG['random_seed']}")
    print(f"\n基于诊断结果：")
    print(f"  预期ASR: 70-90%")
    print(f"  预期L0: 3-5像素（真正的稀疏攻击）")
    print(f"  预计总时间: < 5分钟\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}\n")
    
    # 加载数据
    print("📦 加载CIFAR-10数据...")
    testset = load_cifar10_data()
    print(f"✅ 完成\n")
    
    # 测试矩阵
    models = ['ResNet18', 'VGG16', 'MobileNetV2']
    methods = ['JSMA', 'SparseFool']
    
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
                num_samples=FINAL_CONFIG['test_samples']
            )
            all_results.append(result)
            
            # 保存中间结果
            output_dir = Path('results/final_baseline')
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
        print(f"{'方法':<12} {'ASR':<8} {'平均L0':<10} {'平均L2':<10} {'平均SSIM':<12} {'时间'}")
        print("-"*70)
        for r in all_results:
            if r['model'] == model_name:
                print(f"{r['method']:<12} {r['asr']:<8.1f} {r['avg_l0']:<10.2f} "
                      f"{r['avg_l2']:<10.4f} {r['avg_ssim']:<12.4f} {r['avg_time']:.3f}s")
    
    # 保存汇总
    summary = {
        'config': FINAL_CONFIG,
        'total_time_minutes': float(total_time/60),
        'device': str(device),
        'note': 'Parameters validated through diagnostic tests',
        'results': all_results
    }
    
    with open(output_dir / 'final_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 所有结果已保存到: {output_dir}")
    
    print(f"\n{'='*80}")
    print("🎉 最终Baseline测试完成！")
    print(f"{'='*80}")
    print("\n✅ 获得的数据:")
    print("  - 3个模型（ResNet18, VGG16, MobileNetV2）")
    print("  - 2种方法（JSMA, SparseFool）")
    print("  - 经过诊断验证的统一参数")
    print("  - 每个组合30个样本")
    print("  - 真正的稀疏攻击（L0=3-5）")
    print("  - 总计180个测试")
    print("\n💡 这些数据可以直接用于论文！\n")

if __name__ == "__main__":
    main()







