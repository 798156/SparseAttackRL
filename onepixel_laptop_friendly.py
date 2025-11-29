"""
One-Pixel攻击 - 笔记本友好版
每个模型20样本，参数适中，可分批运行
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
import sys

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

def test_single_model(model_name, device, testset, num_samples=20):
    """测试单个模型（笔记本友好版）"""
    print(f"\n{'='*80}")
    print(f"🎯 测试模型: {model_name}")
    print(f"{'='*80}")
    print(f"参数: maxiter=50, popsize=150 (适中配置)")
    print(f"样本数: {num_samples}")
    
    # 加载模型
    print(f"\n📦 加载{model_name}...")
    model = load_model(model_name, device)
    print(f"✅ 模型加载完成")
    
    # 选择样本
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
    estimated_time = len(selected_samples) * 150 / 60
    print(f"⏰ 预计时间: {estimated_time:.0f}-{estimated_time*1.5:.0f}分钟")
    print(f"💡 温馨提示: 确保笔记本散热良好，可随时按Ctrl+C暂停\n")
    
    # 攻击测试
    success_count = 0
    l0_values = []
    l2_values = []
    time_values = []
    results = []
    
    start_time_total = time.time()
    
    for i, (idx, image, label) in enumerate(tqdm(selected_samples, desc=f"{model_name}")):
        start_time = time.time()
        
        try:
            success, adv_image, modified_info = one_pixel_attack(
                image=image,
                label=label,
                model=model,
                max_iter=50,
                pop_size=150  # 适中的参数，平衡效果和时间
            )
            
            attack_time = time.time() - start_time
            time_values.append(attack_time)
            
            if success:
                diff = (adv_image - image).abs()
                l0 = (diff.sum(dim=0) > 0).sum().item()
                l2 = torch.norm(diff).item()
                
                l0_values.append(l0)
                l2_values.append(l2)
                success_count += 1
                
                results.append({
                    'sample_id': int(idx),
                    'success': True,
                    'l0': float(l0),
                    'l2': float(l2),
                    'time': float(attack_time)
                })
                
                # 每5个成功打印一次
                if success_count % 5 == 0:
                    current_asr = success_count / (i+1) * 100
                    avg_time_so_far = np.mean(time_values)
                    remaining_samples = len(selected_samples) - (i+1)
                    remaining_time = remaining_samples * avg_time_so_far / 60
                    print(f"  ✅ 进度: {success_count}/{i+1}, ASR={current_asr:.1f}%, "
                          f"剩余约{remaining_time:.0f}分钟")
            else:
                results.append({
                    'sample_id': int(idx),
                    'success': False,
                    'time': float(attack_time)
                })
        
        except KeyboardInterrupt:
            print(f"\n⚠️  用户中断！已完成 {i}/{len(selected_samples)} 个样本")
            print(f"当前ASR: {success_count}/{i} = {success_count/max(i,1)*100:.1f}%")
            save_partial = input("是否保存当前结果？(y/n): ")
            if save_partial.lower() == 'y':
                break
            else:
                sys.exit(0)
    
    total_time = time.time() - start_time_total
    
    # 统计结果
    completed_samples = len([r for r in results if 'success' in r])
    asr = success_count / completed_samples * 100 if completed_samples > 0 else 0
    avg_l0 = np.mean(l0_values) if l0_values else 0
    avg_l2 = np.mean(l2_values) if l2_values else 0
    avg_time = np.mean(time_values) if time_values else 0
    
    print(f"\n{'='*80}")
    print(f"📊 {model_name} 测试结果")
    print(f"{'='*80}")
    print(f"  完成样本: {completed_samples}/{num_samples}")
    print(f"  ASR: {success_count}/{completed_samples} = {asr:.1f}%")
    print(f"  平均L0: {avg_l0:.2f}")
    print(f"  平均L2: {avg_l2:.4f}")
    print(f"  平均时间: {avg_time:.1f}秒")
    print(f"  总耗时: {total_time/60:.1f}分钟")
    print(f"{'='*80}\n")
    
    result_data = {
        'model': model_name,
        'parameters': {
            'max_iter': 50,
            'pop_size': 150,
            'seed': 'None'
        },
        'asr': float(asr),
        'success_count': success_count,
        'completed_samples': completed_samples,
        'target_samples': num_samples,
        'avg_l0': float(avg_l0),
        'avg_l2': float(avg_l2),
        'avg_time': float(avg_time),
        'total_time_minutes': float(total_time/60),
        'detailed_results': results
    }
    
    # 保存结果
    output_dir = Path('results/onepixel_laptop')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / f'{model_name.lower()}_result.json', 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"💾 结果已保存到: {output_dir / f'{model_name.lower()}_result.json'}\n")
    
    return result_data

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔬 One-Pixel攻击 - 笔记本友好版")
    print("="*80)
    print("\n💡 设计理念:")
    print("  ✅ 每个模型20样本（统计足够）")
    print("  ✅ 参数适中（popsize=150）")
    print("  ✅ 可分批运行（一次一个模型）")
    print("  ✅ 支持中断恢复")
    print("  ✅ 预计1小时/模型\n")
    
    # 选择要测试的模型
    print("请选择要测试的模型:")
    print("  1. ResNet18")
    print("  2. MobileNetV2")
    print("  3. VGG16 (补充10个样本)")
    print("  4. 全部测试（分3批）")
    
    choice = input("\n请输入选项 (1-4): ").strip()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")
    
    testset = load_cifar10_data()
    
    if choice == '1':
        test_single_model('ResNet18', device, testset, num_samples=20)
    elif choice == '2':
        test_single_model('MobileNetV2', device, testset, num_samples=20)
    elif choice == '3':
        test_single_model('VGG16', device, testset, num_samples=10)
    elif choice == '4':
        print("\n⚠️  建议分批运行，每批之间让笔记本休息15分钟")
        confirm = input("确认全部运行？(y/n): ")
        if confirm.lower() == 'y':
            test_single_model('ResNet18', device, testset, num_samples=20)
            print("\n💤 建议让笔记本休息15分钟再继续...")
            input("按回车继续下一个模型...")
            
            test_single_model('MobileNetV2', device, testset, num_samples=20)
            print("\n💤 建议让笔记本休息15分钟再继续...")
            input("按回车继续下一个模型...")
            
            test_single_model('VGG16', device, testset, num_samples=10)
    else:
        print("无效选项")
        return
    
    print(f"\n{'='*80}")
    print("🎉 测试完成！")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()







