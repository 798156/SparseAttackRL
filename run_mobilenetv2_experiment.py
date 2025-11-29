# run_mobilenetv2_experiment.py
"""
运行MobileNetV2完整实验
测试所有攻击方法：JSMA, One-Pixel, SparseFool

预计时间：2-3小时（100样本）
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import time
from datetime import datetime
import os
from skimage.metrics import structural_similarity as ssim_func

# 导入攻击方法
from jsma_attack import jsma_attack
from one_pixel_attack import one_pixel_attack
from sparsefool_attack import sparsefool_attack


def load_mobilenetv2_model(model_path='cifar10_mobilenetv2.pth', device='cuda'):
    """加载MobileNetV2模型"""
    print(f"📦 加载MobileNetV2模型: {model_path}")
    
    model = torchvision.models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    
    print("✅ MobileNetV2模型加载完成")
    return model


def verify_model_accuracy(model, testloader, device='cuda'):
    """验证模型准确率"""
    print("\n🔍 验证模型准确率...")
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    print(f"✅ 模型准确率: {accuracy:.2f}%")
    return accuracy


def test_jsma(model, sample, label, device='cuda'):
    """测试JSMA攻击"""
    try:
        start_time = time.time()
        
        # JSMA返回 (success, adv_image, modified_pixels)
        success, adv_image, modified_pixels = jsma_attack(
            image=sample.squeeze(0).to(device),  # 确保在正确设备上
            label=label,
            model=model,
            max_pixels=10,
            theta=1.0
        )
        elapsed = time.time() - start_time
        
        # 计算L0和L2
        if success:
            l0 = len(modified_pixels)
            # 确保都在CPU上计算
            adv_cpu = adv_image.cpu()
            sample_cpu = sample.squeeze(0).cpu()
            diff = (adv_cpu - sample_cpu).numpy()
            l2 = np.linalg.norm(diff)
            
            # 计算SSIM
            img1 = sample_cpu.numpy().transpose(1, 2, 0)
            img2 = adv_cpu.numpy().transpose(1, 2, 0)
            ssim = ssim_func(img1, img2, multichannel=True, channel_axis=2, data_range=img2.max()-img2.min())
        else:
            l0 = l2 = ssim = 0.0
        
        return {
            'success': success,
            'l0': l0,
            'l2': l2,
            'ssim': ssim,
            'time': elapsed
        }
    except Exception as e:
        print(f"  JSMA错误: {str(e)}")
        return None


def test_onepixel(model, sample, label, device='cuda'):
    """测试One-Pixel攻击"""
    try:
        start_time = time.time()
        
        # One-Pixel返回 (success, adv_image, modified_info)
        success, adv_image, modified_info = one_pixel_attack(
            image=sample.squeeze(0).to(device),  # 确保在正确设备上
            label=label,
            model=model,
            max_iter=75,
            pixels=1
        )
        elapsed = time.time() - start_time
        
        # 计算指标
        if success:
            l0 = 1.0  # One-Pixel固定修改1个像素
            # 确保都在CPU上计算
            adv_cpu = adv_image.cpu()
            sample_cpu = sample.squeeze(0).cpu()
            diff = (adv_cpu - sample_cpu).numpy()
            l2 = np.linalg.norm(diff)
            
            img1 = sample_cpu.numpy().transpose(1, 2, 0)
            img2 = adv_cpu.numpy().transpose(1, 2, 0)
            ssim = ssim_func(img1, img2, multichannel=True, channel_axis=2, data_range=img2.max()-img2.min())
        else:
            l0 = l2 = ssim = 0.0
        
        return {
            'success': success,
            'l0': l0,
            'l2': l2,
            'ssim': ssim,
            'time': elapsed
        }
    except Exception as e:
        print(f"  One-Pixel错误: {str(e)}")
        return None


def test_sparsefool(model, sample, label, device='cuda'):
    """测试SparseFool攻击"""
    try:
        start_time = time.time()
        
        # SparseFool返回 (success, adv_image, modified_pixels)
        success, adv_image, modified_pixels = sparsefool_attack(
            image=sample.squeeze(0).to(device),  # 确保在正确设备上
            label=label,
            model=model,
            max_iterations=20,
            lambda_=3.0
        )
        elapsed = time.time() - start_time
        
        # 计算指标
        if success:
            l0 = len(modified_pixels)
            # 确保都在CPU上计算
            adv_cpu = adv_image.cpu()
            sample_cpu = sample.squeeze(0).cpu()
            diff = (adv_cpu - sample_cpu).numpy()
            l2 = np.linalg.norm(diff)
            
            img1 = sample_cpu.numpy().transpose(1, 2, 0)
            img2 = adv_cpu.numpy().transpose(1, 2, 0)
            ssim = ssim_func(img1, img2, multichannel=True, channel_axis=2, data_range=img2.max()-img2.min())
        else:
            l0 = l2 = ssim = 0.0
        
        return {
            'success': success,
            'l0': l0,
            'l2': l2,
            'ssim': ssim,
            'time': elapsed
        }
    except Exception as e:
        print(f"  SparseFool错误: {str(e)}")
        return None


def main():
    print("=" * 80)
    print("🚀 MobileNetV2 完整实验")
    print("=" * 80)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载模型
    model = load_mobilenetv2_model(device=device)
    
    # 加载CIFAR-10测试集
    print("\n📦 加载CIFAR-10数据...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    
    # 验证准确率
    accuracy = verify_model_accuracy(model, testloader, device)
    
    # 找到正确分类的样本
    print("\n🔍 找到正确分类的样本...")
    correct_indices = []
    
    with torch.no_grad():
        for idx in range(len(testset)):
            sample, label = testset[idx]
            output = model(sample.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
            
            if pred == label:
                correct_indices.append(idx)
            
            if len(correct_indices) >= 100:
                break
    
    print(f"✅ 找到{len(correct_indices)}个正确分类的样本")
    
    # 运行攻击
    print("\n" + "=" * 80)
    print("🎯 开始攻击测试")
    print("=" * 80)
    
    # 定义攻击方法
    attack_methods = {
        'JSMA': test_jsma,
        'One-Pixel': test_onepixel,
        'SparseFool': test_sparsefool,
    }
    
    # 存储结果
    results = {name: [] for name in attack_methods.keys()}
    
    start_time = time.time()
    
    # 对每个样本测试所有攻击
    for idx, sample_idx in enumerate(correct_indices):
        sample, label = testset[sample_idx]
        sample = sample.unsqueeze(0)  # 添加batch维度
        
        print(f"\n[{idx+1}/{len(correct_indices)}] 样本 {sample_idx}, 标签 {label}")
        
        for method_name, method_func in attack_methods.items():
            print(f"  测试 {method_name}...", end=' ')
            result = method_func(model, sample, label, device)
            
            if result is not None:
                results[method_name].append(result)
                status = "✅" if result['success'] else "❌"
                print(f"{status} L0={result['l0']:.2f}, Time={result['time']:.3f}s")
            else:
                print("⚠️  跳过")
    
    total_time = time.time() - start_time
    
    # 统计结果
    print("\n" + "=" * 80)
    print("📊 实验结果统计")
    print("=" * 80 + "\n")
    
    summary = {}
    detailed = {}
    
    for method_name, method_results in results.items():
        if not method_results:
            print(f"{method_name}: 无结果")
            continue
        
        successes = [r for r in method_results if r['success']]
        asr = len(successes) / len(method_results) * 100
        
        if successes:
            avg_l0 = np.mean([r['l0'] for r in successes])
            avg_l2 = np.mean([r['l2'] for r in successes])
            avg_ssim = np.mean([r['ssim'] for r in successes])
            avg_time = np.mean([r['time'] for r in method_results])
        else:
            avg_l0 = avg_l2 = avg_ssim = avg_time = 0.0
        
        summary[method_name] = {
            'ASR': float(asr),
            'L0': float(avg_l0),
            'L2': float(avg_l2),
            'SSIM': float(avg_ssim),
            'Time': float(avg_time)
        }
        
        # 转换detailed中的numpy类型
        detailed[method_name] = [
            {
                'success': r['success'],
                'l0': float(r['l0']),
                'l2': float(r['l2']),
                'ssim': float(r['ssim']),
                'time': float(r['time'])
            }
            for r in method_results
        ]
        
        print(f"{method_name}:")
        print(f"  ASR:  {asr:.1f}%")
        print(f"  L0:   {avg_l0:.2f}")
        print(f"  L2:   {avg_l2:.4f}")
        print(f"  SSIM: {avg_ssim:.4f}")
        print(f"  Time: {avg_time:.3f}s\n")
    
    # 保存结果
    output_dir = 'results/week1_day5'
    os.makedirs(output_dir, exist_ok=True)
    
    summary_path = os.path.join(output_dir, 'mobilenetv2_summary.json')
    detailed_path = os.path.join(output_dir, 'mobilenetv2_detailed.json')
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(detailed_path, 'w') as f:
        json.dump(detailed, f, indent=2)
    
    print(f"📁 结果已保存到: {summary_path}")
    print(f"📁 详细结果已保存到: {detailed_path}")
    
    # 总结
    print("\n" + "=" * 80)
    print("🎉 实验完成！")
    print(f"⏱️  总耗时: {total_time/60:.1f}分钟")
    print(f"📊 测试样本数: {len(correct_indices)}")
    print(f"🎯 MobileNetV2准确率: {accuracy:.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    main()

