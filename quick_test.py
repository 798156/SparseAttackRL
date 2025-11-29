"""
快速测试所有攻击方法
使用20个样本快速验证所有方法是否正常工作
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import time

from target_model import load_target_model
from jsma_attack import jsma_attack
from one_pixel_attack import one_pixel_attack
from sparsefool_attack import sparsefool_attack_simple
from evaluation_metrics import compute_l0_norm, compute_l2_norm

# 尝试导入RL方法
try:
    from sparse_attack_env_v2 import SparseAttackEnvV2
    from stable_baselines3 import PPO
    rl_available = True
except:
    rl_available = False

def quick_test():
    """快速测试所有攻击方法"""
    print("=" * 80)
    print("🧪 快速测试 - 验证所有攻击方法")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载数据和模型
    print("\n📦 加载数据和模型...")
    
    # 加载CIFAR-10测试集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    
    # 加载模型
    model = load_target_model("resnet18")
    model = model.to(device)
    model.eval()
    
    # 测试参数
    num_samples = 20
    max_pixels = 10  # 增加到10，因为新模型更难攻击
    
    print(f"\n测试样本数: {num_samples}")
    print(f"最大修改像素数: {max_pixels}\n")
    
    # 存储结果
    results = {
        'RL V2': {'success': [], 'l0': [], 'time': []},
        'JSMA': {'success': [], 'l0': [], 'time': []},
        'One-Pixel': {'success': [], 'l0': [], 'time': []},
        'SparseFool': {'success': [], 'l0': [], 'time': []},
    }
    
    # 收集正确分类的样本
    correct_samples = []
    for idx, (images, labels) in enumerate(testloader):
        if len(correct_samples) >= num_samples:
            break
        
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            pred = outputs.argmax(dim=1).item()
            
            if pred == labels.item():
                correct_samples.append((images[0], labels.item()))
    
    print(f"✅ 收集到 {len(correct_samples)} 个正确分类的样本\n")
    
    if len(correct_samples) < num_samples:
        print(f"⚠️ 警告：只找到 {len(correct_samples)} 个正确分类样本，继续测试...\n")
        num_samples = len(correct_samples)
    
    # 测试每个样本
    for idx, (image, label) in enumerate(tqdm(correct_samples, desc="测试进度")):
        image = image.to(device)
        
        # 1. RL V2
        if rl_available:
            try:
                start = time.time()
                env = SparseAttackEnvV2(model, max_steps=max_pixels, device=device)
                agent_path = "results/full_experiments/models/agent_cifar10.zip"
                
                try:
                    agent = PPO.load(agent_path)
                    obs, _ = env.reset(image.unsqueeze(0), label)
                    done = False
                    
                    for _ in range(max_pixels):
                        action, _ = agent.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        if done:
                            break
                    
                    success = info.get('attack_success', False)
                    adv_image = info.get('adv_image', image)
                    elapsed = time.time() - start
                    
                    results['RL V2']['success'].append(success)
                    results['RL V2']['time'].append(elapsed)
                    if success:
                        l0 = compute_l0_norm(image.cpu(), adv_image.cpu())
                        results['RL V2']['l0'].append(l0)
                
                except Exception as e:
                    # 如果找不到模型，跳过
                    results['RL V2']['success'].append(False)
                    results['RL V2']['time'].append(0)
            except Exception as e:
                results['RL V2']['success'].append(False)
                results['RL V2']['time'].append(0)
        
        # 2. JSMA（增加theta到5.0）
        start = time.time()
        try:
            success, adv, pixels = jsma_attack(image, label, model, max_pixels=max_pixels, theta=5.0)
            elapsed = time.time() - start
            
            results['JSMA']['success'].append(success)
            results['JSMA']['time'].append(elapsed)
            if success:
                l0 = compute_l0_norm(image.cpu(), adv.cpu())
                results['JSMA']['l0'].append(l0)
        except Exception as e:
            results['JSMA']['success'].append(False)
            results['JSMA']['time'].append(0)
        
        # 3. One-Pixel（增加迭代次数到200）
        start = time.time()
        try:
            success, params = one_pixel_attack(image, label, model, max_iter=200)
            elapsed = time.time() - start
            
            results['One-Pixel']['success'].append(success)
            results['One-Pixel']['time'].append(elapsed)
            if success:
                results['One-Pixel']['l0'].append(1)  # One-Pixel固定为1
        except Exception as e:
            results['One-Pixel']['success'].append(False)
            results['One-Pixel']['time'].append(0)
        
        # 4. SparseFool
        start = time.time()
        try:
            success, adv, pixels = sparsefool_attack_simple(image, label, model, max_pixels=max_pixels)
            elapsed = time.time() - start
            
            results['SparseFool']['success'].append(success)
            results['SparseFool']['time'].append(elapsed)
            if success:
                l0 = compute_l0_norm(image.cpu(), adv.cpu())
                results['SparseFool']['l0'].append(l0)
        except Exception as e:
            results['SparseFool']['success'].append(False)
            results['SparseFool']['time'].append(0)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("📊 快速测试结果")
    print("=" * 80)
    print(f"\n{'方法':<20} {'ASR (%)':<12} {'平均L0':<12} {'平均时间 (s)':<15}")
    print("-" * 80)
    
    for method, data in results.items():
        if not data['success']:  # 跳过没有数据的方法
            continue
        
        asr = np.mean(data['success']) * 100 if data['success'] else 0
        avg_l0 = np.mean(data['l0']) if data['l0'] else 0
        avg_time = np.mean(data['time']) if data['time'] else 0
        
        print(f"{method:<20} {asr:<12.1f} {avg_l0:<12.2f} {avg_time:<15.3f}")
    
    print("\n" + "=" * 80)
    print("💡 结果分析")
    print("=" * 80)
    
    # 检查异常
    issues = []
    
    for method, data in results.items():
        if not data['success']:
            continue
        
        asr = np.mean(data['success']) * 100
        avg_l0 = np.mean(data['l0']) if data['l0'] else 0
        
        # 检查ASR过低
        if asr < 30:
            issues.append(f"⚠️  {method}: ASR太低 ({asr:.1f}%)，可能需要调整参数")
        
        # 检查L0异常
        if avg_l0 == 0 and asr > 0:
            issues.append(f"⚠️  {method}: L0=0但ASR>0，可能有统计问题")
        
        # 检查L0过大
        if avg_l0 > max_pixels * 2:
            issues.append(f"⚠️  {method}: L0={avg_l0:.1f}超过预期，检查实现")
    
    if issues:
        print("\n发现问题：")
        for issue in issues:
            print(issue)
    else:
        print("\n✅ 所有方法工作正常！")
        print("建议：运行完整实验 (run_full_experiments.py)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    quick_test()

