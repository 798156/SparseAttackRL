"""
One-Pixel攻击 - 快速验证版（10样本/模型）
快速确认趋势，完整测试交给服务器
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
    
    model.to(device)
    model.eval()
    return model

def quick_test_model(model_name, device, testset, num_samples=10):
    """快速测试（10样本）"""
    print(f"\n{'='*80}")
    print(f"⚡ 快速验证: {model_name}")
    print(f"{'='*80}")
    
    model = load_model(model_name, device)
    
    # 选择样本
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
    print(f"⏰ 预计时间: 8-15分钟\n")
    
    # 攻击
    success_count = 0
    results = []
    
    for i, (idx, image, label) in enumerate(tqdm(selected_samples, desc=f"{model_name}")):
        start_time = time.time()
        
        success, adv_image, modified_info = one_pixel_attack(
            image=image,
            label=label,
            model=model,
            max_iter=50,
            pop_size=200
        )
        
        attack_time = time.time() - start_time
        
        if success:
            diff = (adv_image - image).abs()
            l0 = (diff.sum(dim=0) > 0).sum().item()
            success_count += 1
            results.append({'sample_id': int(idx), 'success': True, 'l0': float(l0)})
            print(f"  ✅ 样本{i+1}: 成功! (总计{success_count}/{i+1})")
        else:
            results.append({'sample_id': int(idx), 'success': False})
    
    asr = success_count / len(selected_samples) * 100
    
    print(f"\n📊 {model_name}: ASR = {asr:.1f}% ({success_count}/{len(selected_samples)})")
    
    return {
        'model': model_name,
        'asr': float(asr),
        'success_count': success_count,
        'total_samples': len(selected_samples),
        'results': results
    }

def main():
    """主函数"""
    print("\n" + "="*80)
    print("⚡ One-Pixel快速验证（10样本/模型）")
    print("="*80)
    print("\n💡 策略:")
    print("  ✅ 每个模型10个样本")
    print("  ✅ 快速验证趋势")
    print("  ✅ 预计总时间: 30-45分钟")
    print("  ✅ 完整测试交给服务器\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}\n")
    
    testset = load_cifar10_data()
    
    # 测试3个模型
    models = ['ResNet18', 'MobileNetV2', 'VGG16']
    all_results = []
    
    start_time_total = time.time()
    
    for model_name in models:
        result = quick_test_model(model_name, device, testset, num_samples=10)
        all_results.append(result)
    
    total_time = time.time() - start_time_total
    
    # 汇总
    print(f"\n{'='*80}")
    print("📊 快速验证汇总")
    print(f"{'='*80}")
    print(f"总耗时: {total_time/60:.1f}分钟\n")
    
    print(f"{'模型':<15} {'ASR (10样本)'}")
    print("-"*40)
    for r in all_results:
        print(f"{r['model']:<15} {r['asr']:.1f}%")
    
    # 排序
    print(f"\n🏆 脆弱性排序（ASR从高到低）:")
    sorted_results = sorted(all_results, key=lambda x: x['asr'], reverse=True)
    for i, r in enumerate(sorted_results, 1):
        print(f"  {i}. {r['model']}: {r['asr']:.1f}%")
    
    # 保存
    output_dir = Path('results/onepixel_quick_verify')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    summary = {
        'test_type': 'quick_verification',
        'samples_per_model': 10,
        'total_time_minutes': float(total_time/60),
        'models': all_results
    }
    
    with open(output_dir / 'quick_verify_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 结果已保存到: {output_dir}")
    
    print(f"\n{'='*80}")
    print("💡 下一步建议:")
    print(f"{'='*80}")
    print("基于这个快速验证结果：")
    print("  1. 如果趋势明确 → 部署服务器做100样本完整测试")
    print("  2. 服务器配置：maxiter=50, popsize=200, samples=100")
    print("  3. 预计服务器时间：2-3小时/模型")
    print("  4. 后台运行，明天早上看结果")
    
    print(f"\n🎉 快速验证完成！")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()







