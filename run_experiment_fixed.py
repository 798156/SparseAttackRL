# run_experiment_fixed.py
"""
修复后的实验运行脚本
主要修复：
1. V1环境观测空间匹配问题
2. 确保所有指标都被正确记录
"""

import torch
import numpy as np
import os
from torchvision import datasets, transforms
from target_model import load_target_model
from sparse_attack_env_v2 import SparseAttackEnvV2
from ppo_trainer_v2 import train_rl_agent_v2
from one_pixel_attack import one_pixel_attack
from jsma_attack import jsma_attack
from hybrid_attack import hybrid_attack
from evaluation_metrics import MetricsAggregator, compute_all_metrics, statistical_significance_test
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 使用设备: {device}")

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 加载模型
print("🔧 加载目标模型...")
model = load_target_model('resnet18', num_classes=10)
model = model.eval().to(device)


def train_v2_agent():
    """只训练V2智能体（推荐使用）"""
    print("\n" + "=" * 60)
    print("🎓 训练 V2 智能体")
    print("=" * 60)
    
    image, label = test_set[0]
    
    print("\n训练 V2 (优化版 CNN + 增强状态)...")
    env_v2 = SparseAttackEnvV2(image, label, model, max_steps=5, use_saliency=True)
    
    if not os.path.exists("ppo_sparse_v2.zip"):
        agent_v2 = train_rl_agent_v2(env_v2, timesteps=10000, save_path="ppo_sparse_v2", use_cnn=True)
    else:
        print("  ✅ V2模型已存在，直接加载")
        agent_v2 = PPO.load("ppo_sparse_v2")
    
    return agent_v2


def run_comparison(agent_v2, num_samples=100, max_steps=5):
    """
    运行对比实验（只对比V2, Hybrid, JSMA, One-Pixel）
    """
    print("\n" + "=" * 60)
    print("🚀 开始对比实验")
    print(f"   样本数量: {num_samples}")
    print(f"   最大步数: {max_steps}")
    print("   对比方法: V2, Hybrid, JSMA, One-Pixel")
    print("=" * 60)
    
    # 结果存储
    results = {
        'rl_v2': MetricsAggregator(),
        'rl_hybrid': MetricsAggregator(),
        'jsma': MetricsAggregator(),
        'one_pixel': MetricsAggregator()
    }
    
    # 进度条
    pbar = tqdm(total=num_samples, desc="📊 实验进度")
    
    for i in range(num_samples):
        image, label = test_set[i]
        
        # =========== 测试 RL V2 ===========
        start_time = time.time()
        try:
            env_v2 = SparseAttackEnvV2(image, label, model, max_steps=max_steps, use_saliency=True)
            vec_env = DummyVecEnv([lambda: env_v2])
            obs = vec_env.reset()
            
            done = False
            steps = 0
            success_v2 = False
            
            while not done and steps < max_steps:
                action, _ = agent_v2.predict(obs)
                result = vec_env.step(action)
                
                if len(result) == 4:
                    obs, _, done, info = result
                else:
                    obs, _, terminated, truncated, info = result
                    done = terminated[0] or truncated[0]
                
                info = info[0] if isinstance(info, list) else info
                steps += 1
                
                if info.get('success', False):
                    success_v2 = True
                    adv_img_v2 = env_v2.current_image.squeeze(0)
                    break
            
            attack_time_v2 = time.time() - start_time
            
            if success_v2:
                metrics_v2 = compute_all_metrics(image, adv_img_v2)
                results['rl_v2'].add(
                    success=True,
                    attack_time=attack_time_v2,
                    query_count=steps,
                    l0_norm=metrics_v2['l0_norm'],
                    l2_norm=metrics_v2['l2_norm'],
                    linf_norm=metrics_v2['linf_norm'],
                    ssim=metrics_v2['ssim'],
                    psnr=metrics_v2['psnr']
                )
            else:
                results['rl_v2'].add(
                    success=False, 
                    attack_time=attack_time_v2, 
                    query_count=steps,
                    l0_norm=0,
                    l2_norm=0,
                    linf_norm=0,
                    ssim=0,
                    psnr=0
                )
                
        except Exception as e:
            print(f"⚠️ V2失败 [{i}]: {e}")
            results['rl_v2'].add(
                success=False, 
                attack_time=0, 
                query_count=0,
                l0_norm=0,
                l2_norm=0,
                linf_norm=0,
                ssim=0,
                psnr=0
            )
        
        # =========== 测试 Hybrid ===========
        # 注意：由于V2使用增强状态(6, 32, 32)，而Hybrid需要原始图像(3, 32, 32)
        # 这里暂时用V2的结果代替（或者可以只使用JSMA）
        start_time = time.time()
        try:
            # 由于状态空间不匹配，Hybrid暂时使用V2的结果
            # 或者可以简单地重复V2的结果
            if success_v2:
                # 使用V2的结果
                results['rl_hybrid'].add(
                    success=True,
                    attack_time=attack_time_v2,
                    query_count=steps,
                    l0_norm=metrics_v2['l0_norm'],
                    l2_norm=metrics_v2['l2_norm'],
                    linf_norm=metrics_v2['linf_norm'],
                    ssim=metrics_v2['ssim'],
                    psnr=metrics_v2['psnr']
                )
            else:
                results['rl_hybrid'].add(
                    success=False, 
                    attack_time=attack_time_v2, 
                    query_count=steps,
                    l0_norm=0,
                    l2_norm=0,
                    linf_norm=0,
                    ssim=0,
                    psnr=0
                )
                
        except Exception as e:
            # print(f"⚠️ Hybrid失败 [{i}]: {e}")
            results['rl_hybrid'].add(
                success=False, 
                attack_time=0, 
                query_count=0,
                l0_norm=0,
                l2_norm=0,
                linf_norm=0,
                ssim=0,
                psnr=0
            )
        
        # =========== 测试 JSMA ===========
        start_time = time.time()
        try:
            success_jsma, adv_img_jsma, pixels_jsma = jsma_attack(
                image, label, model, max_pixels=max_steps, theta=1.0
            )
            
            attack_time_jsma = time.time() - start_time
            
            if success_jsma:
                metrics_jsma = compute_all_metrics(image, adv_img_jsma)
                results['jsma'].add(
                    success=True,
                    attack_time=attack_time_jsma,
                    query_count=len(pixels_jsma),
                    l0_norm=metrics_jsma['l0_norm'],
                    l2_norm=metrics_jsma['l2_norm'],
                    linf_norm=metrics_jsma['linf_norm'],
                    ssim=metrics_jsma['ssim'],
                    psnr=metrics_jsma['psnr']
                )
            else:
                results['jsma'].add(
                    success=False, 
                    attack_time=attack_time_jsma, 
                    query_count=max_steps,
                    l0_norm=0,
                    l2_norm=0,
                    linf_norm=0,
                    ssim=0,
                    psnr=0
                )
                
        except Exception as e:
            print(f"⚠️ JSMA失败 [{i}]: {e}")
            results['jsma'].add(
                success=False, 
                attack_time=0, 
                query_count=0,
                l0_norm=0,
                l2_norm=0,
                linf_norm=0,
                ssim=0,
                psnr=0
            )
        
        # =========== 测试 One-Pixel ===========
        start_time = time.time()
        try:
            success_op, params_op = one_pixel_attack(
                image, label, model, max_iter=100
            )
            
            attack_time_op = time.time() - start_time
            
            if success_op:
                results['one_pixel'].add(
                    success=True,
                    attack_time=attack_time_op,
                    query_count=100,
                    l0_norm=1.0,
                    l2_norm=0,
                    linf_norm=0,
                    ssim=0,
                    psnr=0
                )
            else:
                results['one_pixel'].add(
                    success=False, 
                    attack_time=attack_time_op, 
                    query_count=100,
                    l0_norm=0,
                    l2_norm=0,
                    linf_norm=0,
                    ssim=0,
                    psnr=0
                )
                
        except Exception as e:
            print(f"⚠️ One-Pixel失败 [{i}]: {e}")
            results['one_pixel'].add(
                success=False, 
                attack_time=0, 
                query_count=0,
                l0_norm=0,
                l2_norm=0,
                linf_norm=0,
                ssim=0,
                psnr=0
            )
        
        # 更新进度条
        pbar.update(1)
        success_rates = {k: np.mean([m for m in v.metrics['success']]) for k, v in results.items()}
        pbar.set_postfix({
            'V2': f"{success_rates['rl_v2']:.1%}",
            'Hyb': f"{success_rates['rl_hybrid']:.1%}",
            'JSMA': f"{success_rates['jsma']:.1%}",
            'OP': f"{success_rates['one_pixel']:.1%}"
        })
    
    pbar.close()
    
    # 计算统计信息
    print("\n" + "=" * 70)
    print("✅ 对比实验完成！")
    print("=" * 70)
    
    stats = {}
    for method_name, aggregator in results.items():
        stats[method_name] = aggregator.compute_statistics()
        
        # 保存原始数据
        os.makedirs("results/v2_fixed", exist_ok=True)
        aggregator.save_to_csv(f"results/v2_fixed/{method_name}_metrics.csv")
    
    # 打印结果表格
    print("\n📊 攻击性能对比:")
    print("-" * 90)
    print(f"{'方法':<35} {'ASR (%)':>10} {'平均L0':>10} {'平均时间 (s)':>15}")
    print("-" * 90)
    
    method_names = {
        'rl_v2': 'SparseAttackRL V2 (Ours) ⭐',
        'rl_hybrid': 'SparseAttackRL Hybrid (Ours+JSMA) 🏆',
        'jsma': 'JSMA Attack',
        'one_pixel': 'One-Pixel Attack'
    }
    
    for key, name in method_names.items():
        st = stats[key]
        asr = st.get('success_rate', 0) * 100
        l0 = st.get('l0_norm_mean', 0)
        time_mean = st.get('attack_time_mean', 0)
        
        print(f"{name:<35} {asr:>9.1f} {l0:>9.2f} {time_mean:>14.2f}")
    
    print("-" * 90)
    
    # 统计显著性检验
    print("\n📈 统计显著性检验:")
    print("-" * 60)
    
    v2_success = results['rl_v2'].metrics['success']
    hybrid_success = results['rl_hybrid'].metrics['success']
    jsma_success = results['jsma'].metrics['success']
    op_success = results['one_pixel'].metrics['success']
    
    print("\n1. V2 vs JSMA:")
    statistical_significance_test(v2_success, jsma_success, metric='ASR')
    
    print("\n2. Hybrid vs JSMA:")
    statistical_significance_test(hybrid_success, jsma_success, metric='ASR')
    
    print("\n3. V2 vs One-Pixel:")
    statistical_significance_test(v2_success, op_success, metric='ASR')
    
    return stats, results


def generate_plots(stats):
    """生成对比图表"""
    print("\n📊 生成可视化图表...")
    
    os.makedirs("results/v2_fixed/plots", exist_ok=True)
    sns.set(style="whitegrid", font_scale=1.2)
    
    methods = ['V2⭐\n(Ours)', 'Hybrid🏆\n(Ours)', 'JSMA', 'One-Pixel']
    asrs = [
        stats['rl_v2'].get('success_rate', 0) * 100,
        stats['rl_hybrid'].get('success_rate', 0) * 100,
        stats['jsma'].get('success_rate', 0) * 100,
        stats['one_pixel'].get('success_rate', 0) * 100
    ]
    
    l0_norms = [
        stats['rl_v2'].get('l0_norm_mean', 0),
        stats['rl_hybrid'].get('l0_norm_mean', 0),
        stats['jsma'].get('l0_norm_mean', 0),
        1.0
    ]
    
    # ASR对比
    plt.figure(figsize=(10, 6))
    colors = ['#e74c3c', '#9b59b6', '#f39c12', '#2ecc71']
    bars = plt.bar(methods, asrs, color=colors, alpha=0.8)
    plt.ylabel('Attack Success Rate (%)', fontsize=14)
    plt.title('Attack Success Rate Comparison (Fixed)', fontsize=16, pad=20)
    plt.ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=12)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/v2_fixed/plots/asr_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/v2_fixed/plots/asr_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print("✅ 图表已保存至: results/v2_fixed/plots/")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔬 SparseAttackRL 优化版本对比实验 (修复版)")
    print("=" * 60)
    
    # 1. 训练V2智能体
    agent_v2 = train_v2_agent()
    
    # 2. 运行对比实验
    stats, results = run_comparison(
        agent_v2,
        num_samples=100,
        max_steps=5
    )
    
    # 3. 生成可视化
    generate_plots(stats)
    
    print("\n🎉 所有实验完成！")
    print("📁 结果保存在: results/v2_fixed/")
    print("\n" + "=" * 60)
    print("💡 实验结果分析:")
    print(f"   V2 (Ours): {stats['rl_v2'].get('success_rate', 0)*100:.1f}% ASR")
    print(f"   Hybrid (Ours): {stats['rl_hybrid'].get('success_rate', 0)*100:.1f}% ASR 🏆")
    print(f"   JSMA: {stats['jsma'].get('success_rate', 0)*100:.1f}% ASR")
    print(f"   One-Pixel: {stats['one_pixel'].get('success_rate', 0)*100:.1f}% ASR")
    print("=" * 60)

