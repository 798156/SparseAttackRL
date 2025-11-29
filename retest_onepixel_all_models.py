"""
One-Pixel攻击 - 重新测试所有模型
使用统一的正确参数，获得可靠的对比数据
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

from one_pixel_attack import one_pixel_attack

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
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.to(device)
    model.eval()
    return model

def test_model(model_name, device, testset, num_samples=50, max_iter=50, pop_size=200):
    """测试单个模型"""
    print(f"\n{'='*80}")
    print(f"🎯 测试模型: {model_name}")
    print(f"{'='*80}")
    print(f"参数: maxiter={max_iter}, popsize={pop_size}, seed=None (随机)")
    print(f"样本数: {num_samples}")
    
    # 加载模型
    print(f"\n📦 加载{model_name}...")
    model = load_model(model_name, device)
    print(f"✅ 模型加载完成")
    
    # 选择正确分类的样本
    print(f"\n📊 选择{num_samples}个正确分类的样本...")
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
    
    print(f"✅ 选择了 {len(selected_samples)} 个样本")
    
    # 预计时间
    estimated_time = len(selected_samples) * max_iter * pop_size / 20000
    print(f"⏰ 预计时间: {estimated_time:.1f}-{estimated_time*2:.1f}分钟\n")
    
    # 执行攻击
    success_count = 0
    l0_values = []
    l2_values = []
    time_values = []
    results = []
    
    start_time_total = time.time()
    
    for i, (idx, image, label) in enumerate(tqdm(selected_samples, desc=f"{model_name}攻击进度")):
        start_time = time.time()
        
        success, adv_image, modified_info = one_pixel_attack(
            image=image,
            label=label,
            model=model,
            max_iter=max_iter,
            pop_size=pop_size
        )
        
        attack_time = time.time() - start_time
        time_values.append(attack_time)
        
        if success:
            # 计算L0
            diff = (adv_image - image).abs()
            modified_pixels = (diff.sum(dim=0) > 0).sum().item()
            l0_values.append(modified_pixels)
            
            # 计算L2
            l2_norm = torch.norm(diff).item()
            l2_values.append(l2_norm)
            
            success_count += 1
            
            results.append({
                'sample_id': int(idx),
                'success': True,
                'l0': float(modified_pixels),
                'l2': float(l2_norm),
                'time': float(attack_time),
                'modified_pixel': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                                  for k, v in modified_info.items()}
            })
            
            # 每10个成功样本打印一次进度
            if success_count % 5 == 0:
                print(f"  ✅ 已成功: {success_count}/{i+1}, 当前ASR={success_count/(i+1)*100:.1f}%")
        else:
            results.append({
                'sample_id': int(idx),
                'success': False,
                'time': float(attack_time)
            })
    
    total_time = time.time() - start_time_total
    
    # 统计结果
    asr = success_count / len(selected_samples) * 100
    avg_l0 = np.mean(l0_values) if l0_values else 0
    avg_l2 = np.mean(l2_values) if l2_values else 0
    avg_time = np.mean(time_values)
    
    print(f"\n{'='*80}")
    print(f"📊 {model_name} 测试结果")
    print(f"{'='*80}")
    print(f"  ASR: {success_count}/{len(selected_samples)} = {asr:.1f}%")
    print(f"  平均L0: {avg_l0:.2f}")
    print(f"  平均L2: {avg_l2:.4f}")
    print(f"  平均时间: {avg_time:.2f}秒")
    print(f"  总耗时: {total_time/60:.1f}分钟")
    print(f"{'='*80}\n")
    
    return {
        'model': model_name,
        'max_iter': max_iter,
        'pop_size': pop_size,
        'asr': float(asr),
        'success_count': success_count,
        'total_samples': len(selected_samples),
        'avg_l0': float(avg_l0),
        'avg_l2': float(avg_l2),
        'avg_time': float(avg_time),
        'total_time': float(total_time),
        'detailed_results': results
    }

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔬 One-Pixel攻击 - 重新测试所有模型")
    print("="*80)
    print("\n💡 统一测试参数:")
    print("  ✅ maxiter=50")
    print("  ✅ popsize=200")
    print("  ✅ seed=None (随机探索)")
    print("  ✅ 每个模型50个样本")
    print("  ✅ 预计总时间: 30-60分钟\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}\n")
    
    # 加载数据
    print("📦 加载CIFAR-10数据...")
    testset = load_cifar10_data()
    print(f"✅ 数据加载完成\n")
    
    # 测试所有模型
    models = ['ResNet18', 'MobileNetV2', 'VGG16']
    all_results = []
    
    for model_name in models:
        result = test_model(
            model_name=model_name,
            device=device,
            testset=testset,
            num_samples=50,
            max_iter=50,
            pop_size=200
        )
        all_results.append(result)
        
        # 每个模型测试完后保存一次（防止意外中断）
        output_dir = Path('results/onepixel_retest')
        output_dir.mkdir(exist_ok=True, parents=True)
        
        with open(output_dir / f'{model_name.lower()}_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"💾 {model_name}结果已保存\n")
    
    # 汇总对比
    print("\n" + "="*80)
    print("📊 所有模型对比汇总")
    print("="*80)
    print(f"\n{'模型':<15} {'ASR':<10} {'平均L0':<10} {'平均L2':<12} {'平均时间'}")
    print("-"*80)
    for r in all_results:
        print(f"{r['model']:<15} {r['asr']:<10.1f} {r['avg_l0']:<10.2f} "
              f"{r['avg_l2']:<12.4f} {r['avg_time']:.2f}s")
    
    # 排序分析
    print(f"\n🏆 按攻击难度排序（ASR从高到低，越高越脆弱）:")
    sorted_results = sorted(all_results, key=lambda x: x['asr'], reverse=True)
    for i, r in enumerate(sorted_results, 1):
        print(f"  {i}. {r['model']}: {r['asr']:.1f}% ASR")
    
    # 保存汇总
    summary = {
        'test_parameters': {
            'max_iter': 50,
            'pop_size': 200,
            'seed': 'None (random)',
            'samples_per_model': 50
        },
        'models': all_results,
        'ranking': [r['model'] for r in sorted_results]
    }
    
    with open(output_dir / 'all_models_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 汇总结果已保存到: {output_dir / 'all_models_summary.json'}")
    
    print(f"\n{'='*80}")
    print("🎉 所有模型重新测试完成！")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()







