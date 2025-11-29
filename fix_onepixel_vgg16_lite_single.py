"""
One-Pixel攻击修复 - 只跑配置1（10个样本）
安全备用脚本，如果当前脚本跑完配置1后要手动停止
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

def load_model():
    """加载VGG16模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = torchvision.models.vgg16(weights=None)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 10)
    
    model.load_state_dict(torch.load('cifar10_vgg16.pth', map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    
    return model, device

def load_test_data():
    """加载测试数据"""
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    return testset

def main():
    """只测试配置1"""
    print("\n" + "="*80)
    print("🔧 One-Pixel攻击 - 仅配置1（10样本）")
    print("="*80)
    
    # 加载
    print("\n📦 加载模型和数据...")
    model, device = load_model()
    testset = load_test_data()
    print(f"✅ 完成\n")
    
    # 只测试轻量配置
    print("="*80)
    print("🧪 轻量配置")
    print("="*80)
    print("参数: maxiter=50, popsize=200, 样本数=10")
    
    # 选择样本
    selected_samples = []
    for idx in range(len(testset)):
        if len(selected_samples) >= 10:
            break
        image, label = testset[idx]
        image_batch = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_batch)
            pred = output.argmax(dim=1).item()
        if pred == label:
            selected_samples.append((idx, image, label))
    
    print(f"✅ 选择了 {len(selected_samples)} 个样本\n")
    
    # 攻击
    success_count = 0
    l0_values = []
    results = []
    
    for i, (idx, image, label) in enumerate(tqdm(selected_samples, desc="攻击进度")):
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
            modified_pixels = (diff.sum(dim=0) > 0).sum().item()
            l0_values.append(modified_pixels)
            success_count += 1
            results.append({
                'sample_id': idx,
                'success': True,
                'l0': modified_pixels,
                'time': attack_time
            })
            print(f"  ✅ 样本{i+1}: 成功! L0={modified_pixels}")
        else:
            results.append({
                'sample_id': idx,
                'success': False,
                'time': attack_time
            })
    
    # 结果
    asr = success_count / len(selected_samples) * 100
    avg_l0 = np.mean(l0_values) if l0_values else 0
    
    print(f"\n{'='*80}")
    print(f"📊 配置1最终结果")
    print(f"{'='*80}")
    print(f"ASR: {asr:.1f}% ({success_count}/{len(selected_samples)})")
    print(f"平均L0: {avg_l0:.2f}")
    
    # 保存
    output_dir = Path('results/onepixel_fix')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    result_data = {
        'config': '轻量配置',
        'max_iter': 50,
        'pop_size': 200,
        'asr': float(asr),
        'success_count': success_count,
        'total_samples': len(selected_samples),
        'avg_l0': float(avg_l0),
        'detailed_results': results
    }
    
    with open(output_dir / 'config1_only.json', 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n✅ 结果已保存")
    print(f"\n{'='*80}")
    print("🎉 配置1完成！")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()







