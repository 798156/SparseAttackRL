# run_simple_comparison.py
"""
简化的对比实验脚本
只对比核心方法：V2 (Ours), JSMA, One-Pixel
避免了Hybrid的状态空间匹配问题
"""

import torch
import numpy as np
import os
from torchvision import datasets, transforms

# 设置matplotlib后端（避免Qt错误）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from target_model import load_target_model
from sparse_attack_env_v2 import SparseAttackEnvV2
from ppo_trainer_v2 import train_rl_agent_v2
from one_pixel_attack import one_pixel_attack
from jsma_attack import jsma_attack
from evaluation_metrics import MetricsAggregator, compute_all_metrics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
import time
from scipy import stats as scipy_stats

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
    """训练V2智能体"""
    print("\n" + "=" * 60)
    print("🎓 训练 V2 智能体")
    print("=" * 60)
    
    image, label = test_set[0]
    env_v2 = SparseAttackEnvV2(image, label, model, max_steps=5, use_saliency=True)
    
    if not os.path.exists("ppo_sparse_v2.zip"):
        agent_v2 = train_rl_agent_v2(env_v2, timesteps=10000, save_path="ppo_sparse_v2", use_cnn=True)
    else:
        print("  ✅ V2模型已存在，直接加载")
        agent_v2 = PPO.load("ppo_sparse_v2")
    
    return agent_v2


def run_comparison(agent_v2, num_samples=100, max_steps=5):
    """运行简化的对比实验"""
    print("\n" + "=" * 60)
    print("🚀 开始对比实验")
    print(f"   样本数量: {num_samples}")
    print(f"   最大步数: {max_steps}")
    print(f"   对比方法: V2 (Ours), JSMA, One-Pixel")
    print("=" * 60)
    
    results = {
        'rl_v2': MetricsAggregator(),
        'jsma': MetricsAggregator(),
        'one_pixel': MetricsAggregator()
    }
    
    pbar = tqdm(total=num_samples, desc="📊 实验进度")
    
    for i in range(num_samples):
        image, label = test_set[i]
        
        # === V2 ===
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
                results['rl_v2'].add(success=False, attack_time=attack_time_v2, query_count=steps,
                                   l0_norm=0, l2_norm=0, linf_norm=0, ssim=0, psnr=0)
        except Exception as e:
            results['rl_v2'].add(success=False, attack_time=0, query_count=0,
                               l0_norm=0, l2_norm=0, linf_norm=0, ssim=0, psnr=0)
        
        # === JSMA ===
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
                results['jsma'].add(success=False, attack_time=attack_time_jsma, query_count=max_steps,
                                  l0_norm=0, l2_norm=0, linf_norm=0, ssim=0, psnr=0)
        except Exception as e:
            results['jsma'].add(success=False, attack_time=0, query_count=0,
                              l0_norm=0, l2_norm=0, linf_norm=0, ssim=0, psnr=0)
        
        # === One-Pixel ===
        start_time = time.time()
        try:
            success_op, params_op = one_pixel_attack(image, label, model, max_iter=100)
            attack_time_op = time.time() - start_time
            
            if success_op:
                results['one_pixel'].add(success=True, attack_time=attack_time_op, query_count=100,
                                       l0_norm=1.0, l2_norm=0, linf_norm=0, ssim=0, psnr=0)
            else:
                results['one_pixel'].add(success=False, attack_time=attack_time_op, query_count=100,
                                       l0_norm=0, l2_norm=0, linf_norm=0, ssim=0, psnr=0)
        except Exception as e:
            results['one_pixel'].add(success=False, attack_time=0, query_count=0,
                                   l0_norm=0, l2_norm=0, linf_norm=0, ssim=0, psnr=0)
        
        # 更新进度条
        pbar.update(1)
        success_rates = {k: np.mean([m for m in v.metrics['success']]) for k, v in results.items()}
        pbar.set_postfix({
            'V2': f"{success_rates['rl_v2']:.1%}",
            'JSMA': f"{success_rates['jsma']:.1%}",
            'OP': f"{success_rates['one_pixel']:.1%}"
        })
    
    pbar.close()
    
    # 计算统计信息
    print("\n" + "=" * 80)
    print("✅ 对比实验完成！")
    print("=" * 80)
    
    stats = {}
    for method_name, aggregator in results.items():
        stats[method_name] = aggregator.compute_statistics()
        os.makedirs("results/final", exist_ok=True)
        aggregator.save_to_csv(f"results/final/{method_name}_metrics.csv")
    
    # 打印结果表格
    print("\n📊 攻击性能对比:")
    print("-" * 90)
    print(f"{'方法':<30} {'ASR (%)':>10} {'平均L0':>10} {'平均L2':>10} {'平均时间 (s)':>15}")
    print("-" * 90)
    
    method_names = {
        'rl_v2': 'SparseAttackRL V2 (Ours) ⭐',
        'jsma': 'JSMA Attack',
        'one_pixel': 'One-Pixel Attack'
    }
    
    for key, name in method_names.items():
        st = stats[key]
        asr = st.get('success_rate', 0) * 100
        l0 = st.get('l0_norm_mean', 0)
        l2 = st.get('l2_norm_mean', 0)
        time_mean = st.get('attack_time_mean', 0)
        
        print(f"{name:<30} {asr:>9.1f} {l0:>9.2f} {l2:>9.2f} {time_mean:>14.2f}")
    
    print("-" * 90)
    
    # 统计显著性检验
    print("\n📈 统计显著性检验:")
    print("-" * 60)
    
    v2_success = np.array(results['rl_v2'].metrics['success'], dtype=float)
    jsma_success = np.array(results['jsma'].metrics['success'], dtype=float)
    op_success = np.array(results['one_pixel'].metrics['success'], dtype=float)
    
    # V2 vs JSMA
    t_stat, p_value = scipy_stats.ttest_rel(v2_success, jsma_success)
    print(f"\n1. V2 vs JSMA:")
    print(f"   t统计量: {t_stat:.4f}")
    print(f"   p值: {p_value:.4f}")
    print(f"   显著性(α=0.05): {'✅ 显著' if p_value < 0.05 else '❌ 不显著'}")
    
    # V2 vs One-Pixel
    t_stat, p_value = scipy_stats.ttest_rel(v2_success, op_success)
    print(f"\n2. V2 vs One-Pixel:")
    print(f"   t统计量: {t_stat:.4f}")
    print(f"   p值: {p_value:.4f}")
    print(f"   显著性(α=0.05): {'✅ 显著' if p_value < 0.05 else '❌ 不显著'}")
    
    return stats, results


def generate_plots(stats):
    """生成对比图表"""
    print("\n📊 生成可视化图表...")
    
    os.makedirs("results/final/plots", exist_ok=True)
    sns.set(style="whitegrid", font_scale=1.3)
    
    methods = ['V2⭐\n(Ours)', 'JSMA', 'One-Pixel']
    asrs = [
        stats['rl_v2'].get('success_rate', 0) * 100,
        stats['jsma'].get('success_rate', 0) * 100,
        stats['one_pixel'].get('success_rate', 0) * 100
    ]
    
    l0_norms = [
        stats['rl_v2'].get('l0_norm_mean', 0),
        stats['jsma'].get('l0_norm_mean', 0),
        1.0
    ]
    
    # ASR对比
    plt.figure(figsize=(10, 6))
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    bars = plt.bar(methods, asrs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.ylabel('Attack Success Rate (%)', fontsize=14)
    plt.title('Attack Success Rate Comparison', fontsize=16, pad=20, weight='bold')
    plt.ylim(0, 105)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=13, weight='bold')
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('results/final/plots/asr_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/final/plots/asr_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    # L0范数对比
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, l0_norms, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.ylabel('Average Modified Pixels (L0 Norm)', fontsize=14)
    plt.title('Sparsity Comparison', fontsize=16, pad=20, weight='bold')
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=13, weight='bold')
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('results/final/plots/l0_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/final/plots/l0_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print("✅ 图表已保存至: results/final/plots/")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔬 SparseAttackRL 核心方法对比实验")
    print("=" * 60)
    
    agent_v2 = train_v2_agent()
    stats, results = run_comparison(agent_v2, num_samples=100, max_steps=5)
    generate_plots(stats)
    
    print("\n🎉 所有实验完成！")
    print("📁 结果保存在: results/final/")
    print("\n" + "=" * 80)
    print("💡 实验结果总结:")
    print(f"   V2 (Ours):  ASR={stats['rl_v2'].get('success_rate', 0)*100:.1f}%, L0={stats['rl_v2'].get('l0_norm_mean', 0):.2f}")
    print(f"   JSMA:       ASR={stats['jsma'].get('success_rate', 0)*100:.1f}%, L0={stats['jsma'].get('l0_norm_mean', 0):.2f}")
    print(f"   One-Pixel:  ASR={stats['one_pixel'].get('success_rate', 0)*100:.1f}%, L0={stats['one_pixel'].get('l0_norm_mean', 0):.2f}")
    print("=" * 80)

