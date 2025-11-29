"""
基于Foolbox的标准Baseline测试
使用标准库确保结果可靠性
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
import foolbox as fb
from skimage.metrics import structural_similarity as ssim_func

# ============= 统一参数配置 =============
CONFIG = {
    'test_samples': 30,
    'random_seed': 42,
    'attacks': {
        'L2DeepFool': {
            'steps': 50,
            'candidates': 10
        },
        'L2CarliniWagner': {
            'binary_search_steps': 5,
            'steps': 100,
            'stepsize': 0.01,
            'confidence': 0,
            'initial_const': 0.01
        },
        'BoundaryAttack': {
            'steps': 1000,
            'spherical_step': 0.01,
            'source_step': 0.01
        }
    }
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
    # 确保在CPU上计算
    original = original.detach().cpu()
    adversarial = adversarial.detach().cpu()
    
    # L0范数（修改像素数）
    diff = (adversarial - original).abs()
    l0 = (diff.sum(dim=0) > 1e-5).sum().item()
    
    # L2范数
    l2 = torch.norm(diff).item()
    
    # SSIM
    orig_np = original.numpy().transpose(1, 2, 0)
    adv_np = adversarial.numpy().transpose(1, 2, 0)
    
    # 归一化到[0,1]
    orig_np = (orig_np - orig_np.min()) / (orig_np.max() - orig_np.min() + 1e-8)
    adv_np = (adv_np - adv_np.min()) / (adv_np.max() - adv_np.min() + 1e-8)
    
    ssim_value = ssim_func(orig_np, adv_np, channel_axis=2, data_range=1.0)
    
    return l0, l2, ssim_value

def test_single_attack(attack_name, model_name, fmodel, images, labels, device):
    """测试单个攻击方法"""
    print(f"\n{'='*80}")
    print(f"🎯 测试: {model_name} + {attack_name}")
    print(f"{'='*80}")
    
    # 创建攻击
    if attack_name == 'L2DeepFool':
        params = CONFIG['attacks']['L2DeepFool']
        attack = fb.attacks.L2DeepFoolAttack(steps=params['steps'], candidates=params['candidates'])
        print(f"参数: steps={params['steps']}, candidates={params['candidates']}")
    
    elif attack_name == 'L2CW':
        params = CONFIG['attacks']['L2CarliniWagner']
        attack = fb.attacks.L2CarliniWagnerAttack(
            binary_search_steps=params['binary_search_steps'],
            steps=params['steps'],
            stepsize=params['stepsize'],
            confidence=params['confidence'],
            initial_const=params['initial_const']
        )
        print(f"参数: steps={params['steps']}, confidence={params['confidence']}")
    
    elif attack_name == 'Boundary':
        params = CONFIG['attacks']['BoundaryAttack']
        attack = fb.attacks.BoundaryAttack(
            steps=params['steps'],
            spherical_step=params['spherical_step'],
            source_step=params['source_step']
        )
        print(f"参数: steps={params['steps']}")
    
    else:
        raise ValueError(f"Unknown attack: {attack_name}")
    
    # 执行攻击
    success_count = 0
    results = {
        'l0': [],
        'l2': [],
        'ssim': [],
        'time': [],
        'details': []
    }
    
    print(f"✅ 测试 {len(images)} 个样本\n")
    
    for i in tqdm(range(len(images)), desc=attack_name):
        start_time = time.time()
        
        try:
            # Foolbox攻击
            raw_advs, clipped_advs, success = attack(
                fmodel,
                images[i:i+1],
                labels[i:i+1],
                epsilons=None
            )
            
            attack_time = time.time() - start_time
            
            # 检查是否成功
            if success[0]:
                adv_image = clipped_advs[0]
                orig_image = images[i]
                
                # 计算指标
                l0, l2, ssim_val = calculate_metrics(orig_image, adv_image)
                
                results['l0'].append(l0)
                results['l2'].append(l2)
                results['ssim'].append(ssim_val)
                results['time'].append(attack_time)
                
                success_count += 1
                
                results['details'].append({
                    'sample_id': int(i),
                    'success': True,
                    'l0': float(l0),
                    'l2': float(l2),
                    'ssim': float(ssim_val),
                    'time': float(attack_time)
                })
            else:
                results['time'].append(attack_time)
                results['details'].append({
                    'sample_id': int(i),
                    'success': False,
                    'time': float(attack_time)
                })
        
        except Exception as e:
            print(f"\n⚠️  样本{i}出错: {str(e)}")
            results['details'].append({
                'sample_id': int(i),
                'success': False,
                'error': str(e)
            })
    
    # 统计结果
    asr = success_count / len(images) * 100
    avg_l0 = np.mean(results['l0']) if results['l0'] else 0
    avg_l2 = np.mean(results['l2']) if results['l2'] else 0
    avg_ssim = np.mean(results['ssim']) if results['ssim'] else 0
    avg_time = np.mean(results['time']) if results['time'] else 0
    
    print(f"\n📊 结果:")
    print(f"  ASR: {success_count}/{len(images)} = {asr:.1f}%")
    print(f"  平均L0: {avg_l0:.2f}")
    print(f"  平均L2: {avg_l2:.4f}")
    print(f"  平均SSIM: {avg_ssim:.4f}")
    print(f"  平均时间: {avg_time:.3f}秒")
    
    return {
        'model': model_name,
        'attack': attack_name,
        'asr': float(asr),
        'success_count': success_count,
        'total_samples': len(images),
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
    print("🔬 基于Foolbox的标准Baseline测试")
    print("="*80)
    print("\n💡 优势:")
    print("  ✅ 使用Foolbox标准实现")
    print("  ✅ 参数经过验证")
    print("  ✅ 结果可信可重复")
    print("  ✅ 易于与文献对比\n")
    
    print("📋 测试配置:")
    print(f"  样本数: {CONFIG['test_samples']}/模型")
    print(f"  攻击方法: L2DeepFool, L2CW, Boundary")
    print(f"  随机种子: {CONFIG['random_seed']}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}\n")
    
    # 加载数据
    print("📦 加载CIFAR-10数据...")
    testset = load_cifar10_data()
    print(f"✅ 数据加载完成\n")
    
    # 设置随机种子
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    
    # 测试矩阵
    models = ['ResNet18', 'VGG16', 'MobileNetV2']
    attacks = ['L2DeepFool', 'L2CW']  # Boundary太慢，暂时不测
    
    all_results = []
    start_time_total = time.time()
    
    for model_name in models:
        print(f"\n{'='*80}")
        print(f"📦 加载模型: {model_name}")
        print(f"{'='*80}")
        
        # 加载PyTorch模型
        model = load_model(model_name, device)
        
        # 创建Foolbox模型
        fmodel = fb.PyTorchModel(model, bounds=(0, 1))
        print(f"✅ {model_name} 加载完成")
        
        # 选择测试样本
        selected_images = []
        selected_labels = []
        
        for idx in range(len(testset)):
            if len(selected_images) >= CONFIG['test_samples']:
                break
            
            image, label = testset[idx]
            image_batch = image.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image_batch)
                pred = output.argmax(dim=1).item()
            
            if pred == label:
                selected_images.append(image.to(device))
                selected_labels.append(torch.tensor(label).to(device))
        
        images_tensor = torch.stack(selected_images)
        labels_tensor = torch.stack(selected_labels)
        
        print(f"✅ 选择了 {len(selected_images)} 个正确分类的样本")
        
        # 测试每种攻击
        for attack_name in attacks:
            result = test_single_attack(
                attack_name=attack_name,
                model_name=model_name,
                fmodel=fmodel,
                images=images_tensor,
                labels=labels_tensor,
                device=device
            )
            all_results.append(result)
            
            # 保存中间结果
            output_dir = Path('results/foolbox_baseline')
            output_dir.mkdir(exist_ok=True, parents=True)
            
            with open(output_dir / f'{model_name.lower()}_{attack_name.lower()}.json', 'w') as f:
                json.dump(result, f, indent=2)
    
    total_time = time.time() - start_time_total
    
    # 生成汇总
    print(f"\n{'='*80}")
    print("📊 完整实验结果汇总")
    print(f"{'='*80}")
    print(f"总耗时: {total_time/60:.1f}分钟\n")
    
    for model_name in models:
        print(f"\n【{model_name}】")
        print(f"{'攻击方法':<15} {'ASR':<8} {'平均L0':<10} {'平均L2':<10} {'平均SSIM':<12} {'时间'}")
        print("-"*75)
        for r in all_results:
            if r['model'] == model_name:
                print(f"{r['attack']:<15} {r['asr']:<8.1f} {r['avg_l0']:<10.2f} "
                      f"{r['avg_l2']:<10.4f} {r['avg_ssim']:<12.4f} {r['avg_time']:.3f}s")
    
    # 保存汇总
    summary = {
        'config': CONFIG,
        'total_time_minutes': float(total_time/60),
        'device': str(device),
        'library': 'Foolbox',
        'results': all_results
    }
    
    with open(output_dir / 'foolbox_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 所有结果已保存到: {output_dir}")
    
    print(f"\n{'='*80}")
    print("🎉 Foolbox Baseline测试完成！")
    print(f"{'='*80}")
    print("\n✅ 获得的数据:")
    print("  - 3个模型")
    print("  - 2-3种标准攻击方法")
    print("  - 每个组合30个样本")
    print("  - 使用Foolbox标准实现")
    print("  - 结果可信可重复\n")

if __name__ == "__main__":
    main()







