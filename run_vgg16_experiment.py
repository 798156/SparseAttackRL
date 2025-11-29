# run_vgg16_experiment.py
"""
运行VGG16完整实验
测试所有攻击方法：RL V1, RL V2, JSMA, One-Pixel, SparseFool

预计时间：2-3小时（100-200样本）
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
from stable_baselines3 import PPO
from sparse_attack_env import SparseAttackEnv
from sparse_attack_env_v2 import SparseAttackEnvV2


def load_vgg16_model(model_path='cifar10_vgg16.pth', device='cuda'):
    """加载VGG16模型"""
    print(f"📦 加载VGG16模型: {model_path}")
    
    model = torchvision.models.vgg16(weights=None)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_ftrs, 10)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print("✅ VGG16模型加载完成")
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
            image=sample.squeeze(0),  # 去掉batch维度
            label=label,
            model=model,
            max_pixels=10,
            theta=1.0
        )
        elapsed = time.time() - start_time
        
        # 计算L0和L2
        if success:
            l0 = len(modified_pixels)
            diff = (adv_image - sample.squeeze(0)).cpu().numpy()
            l2 = np.linalg.norm(diff)
            
            # 计算SSIM
            img1 = sample.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            img2 = adv_image.cpu().numpy().transpose(1, 2, 0)
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
            image=sample.squeeze(0),
            label=label,
            model=model,
            max_iter=75,
            pixels=1
        )
        elapsed = time.time() - start_time
        
        # 计算指标
        if success:
            l0 = 1.0  # One-Pixel固定修改1个像素
            diff = (adv_image - sample.squeeze(0)).cpu().numpy()
            l2 = np.linalg.norm(diff)
            
            img1 = sample.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            img2 = adv_image.cpu().numpy().transpose(1, 2, 0)
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
            image=sample.squeeze(0),
            label=label,
            model=model,
            max_iterations=20,
            lambda_=3.0
        )
        elapsed = time.time() - start_time
        
        # 计算指标
        if success:
            l0 = len(modified_pixels)
            diff = (adv_image - sample.squeeze(0)).cpu().numpy()
            l2 = np.linalg.norm(diff)
            
            img1 = sample.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            img2 = adv_image.cpu().numpy().transpose(1, 2, 0)
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


def test_rl_v1(model, sample, label, device='cuda'):
    """测试RL V1攻击"""
    try:
        if not os.path.exists('ppo_sparse_v1.zip'):
            print("  ⚠️  RL V1模型不存在，跳过")
            return None
        
        start_time = time.time()
        
        # 创建环境 - 参数是 (clean_image, true_label, model, max_steps)
        env = SparseAttackEnv(
            clean_image=sample.squeeze(0),
            true_label=label,
            model=model,
            max_steps=5
        )
        
        # 加载训练好的RL agent
        rl_agent = PPO.load('ppo_sparse_v1', device=device)
        rl_agent.set_env(env)
        
        # 重置环境
        obs, _ = env.reset()
        
        # 执行攻击
        done = False
        step = 0
        
        while not done and step < 10:
            action, _ = rl_agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
        
        elapsed = time.time() - start_time
        
        # 检查是否成功
        with torch.no_grad():
            output = model(env.current_image)
            pred = output.argmax(dim=1).item()
            success = (pred != label)
        
        # 计算指标
        if success:
            l0 = env.current_step
            diff = (env.current_image - env.clean_image).cpu().numpy()
            l2 = np.linalg.norm(diff)
            
            img1 = env.clean_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            img2 = env.current_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
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
        print(f"  RL V1错误: {str(e)}")
        return None


def test_rl_v2(model, sample, label, device='cuda'):
    """测试RL V2攻击"""
    try:
        if not os.path.exists('ppo_sparse_v2.zip'):
            print("  ⚠️  RL V2模型不存在，跳过")
            return None
        
        start_time = time.time()
        
        # 创建环境 - 参数是 (clean_image, true_label, model, max_steps)
        env = SparseAttackEnvV2(
            clean_image=sample.squeeze(0),
            true_label=label,
            model=model,
            max_steps=5,
            use_saliency=True
        )
        
        # 加载训练好的RL agent  
        rl_agent = PPO.load('ppo_sparse_v2', device=device)
        rl_agent.set_env(env)
        
        # 重置环境
        obs, _ = env.reset()
        
        # 执行攻击
        done = False
        step = 0
        
        while not done and step < 10:
            action, _ = rl_agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
        
        elapsed = time.time() - start_time
        
        # 检查是否成功
        with torch.no_grad():
            output = model(env.current_image)
            pred = output.argmax(dim=1).item()
            success = (pred != label)
        
        # 计算指标
        if success:
            l0 = env.current_step
            diff = (env.current_image - env.clean_image).cpu().numpy()
            l2 = np.linalg.norm(diff)
            
            img1 = env.clean_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            img2 = env.current_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
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
        print(f"  RL V2错误: {str(e)}")
        return None


def run_experiments(num_samples=100):
    """运行完整实验"""
    print("=" * 80)
    print("🎯 VGG16完整实验")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 加载VGG16模型
    if not os.path.exists('cifar10_vgg16.pth'):
        print("❌ 错误：VGG16模型不存在！")
        print("请先运行: python train_cifar10_vgg16.py")
        return
    
    model = load_vgg16_model(device=device)
    
    # 加载测试数据
    print("\n📦 加载CIFAR-10测试数据...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # 验证模型准确率
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    accuracy = verify_model_accuracy(model, testloader, device)
    
    if accuracy < 75.0:
        print(f"⚠️  警告：模型准确率较低 ({accuracy:.2f}%)，实验结果可能不理想")
    
    # 选择测试样本（只选择正确分类的样本）
    print(f"\n🔍 选择{num_samples}个测试样本...")
    
    correct_indices = []
    with torch.no_grad():
        for idx in range(len(testset)):
            sample, label = testset[idx]
            sample = sample.unsqueeze(0).to(device)
            output = model(sample)
            pred = output.argmax(dim=1).item()
            
            if pred == label:
                correct_indices.append(idx)
            
            if len(correct_indices) >= num_samples:
                break
    
    print(f"✅ 选择了{len(correct_indices)}个正确分类的样本")
    
    # 准备攻击方法
    attack_methods = {
        'JSMA': test_jsma,
        'One-Pixel': test_onepixel,
        'SparseFool': test_sparsefool,
        'RL-V1': test_rl_v1,
        'RL-V2': test_rl_v2,
    }
    
    # 初始化结果
    results = {method: [] for method in attack_methods.keys()}
    
    # 运行实验
    print("\n" + "=" * 80)
    print("开始攻击测试...")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    for idx, sample_idx in enumerate(correct_indices):
        sample, label = testset[sample_idx]
        sample = sample.unsqueeze(0).to(device)
        
        print(f"[{idx+1}/{len(correct_indices)}] 样本 {sample_idx}, 标签 {label}")
        
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
        
        print(f"{method_name}:")
        print(f"  ASR:  {asr:.1f}%")
        print(f"  L0:   {avg_l0:.2f}")
        print(f"  L2:   {avg_l2:.4f}")
        print(f"  SSIM: {avg_ssim:.4f}")
        print(f"  Time: {avg_time:.3f}s")
        print()
    
    # 保存结果
    output_dir = 'results/week1_day2'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON
    output_file = os.path.join(output_dir, 'vgg16_summary.json')
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📁 结果已保存到: {output_file}")
    
    # 保存详细结果
    detailed_file = os.path.join(output_dir, 'vgg16_detailed.json')
    with open(detailed_file, 'w') as f:
        # 将numpy类型转换为Python原生类型
        serializable_results = {}
        for method, method_results in results.items():
            serializable_results[method] = [
                {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                 for k, v in r.items()}
                for r in method_results
            ]
        json.dump(serializable_results, f, indent=2)
    
    print(f"📁 详细结果已保存到: {detailed_file}")
    
    # 打印总结
    print("\n" + "=" * 80)
    print(f"🎉 实验完成！")
    print(f"⏱️  总耗时: {total_time/60:.1f}分钟")
    print(f"📊 测试样本数: {len(correct_indices)}")
    print(f"🎯 VGG16准确率: {accuracy:.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='VGG16完整实验')
    parser.add_argument('--num_samples', type=int, default=100, help='测试样本数')
    args = parser.parse_args()
    
    run_experiments(num_samples=args.num_samples)

