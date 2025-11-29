"""
快速保存Day 1的实验结果
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 从终端输出提取的数据
results = {
    'JSMA': {
        'ASR': 55.0,
        'L0': 4.80,
        'L2': 6.8784,
        'SSIM': 0.9227,
        'Time': 0.562
    },
    'One-Pixel': {
        'ASR': 16.0,
        'L0': 1.00,
        'L2': 0.0000,
        'SSIM': 1.0000,
        'Time': 12.606
    },
    'SparseFool': {
        'ASR': 47.0,
        'L0': 4.74,
        'L2': 5.9634,
        'SSIM': 0.9551,
        'Time': 0.151
    }
}

# 创建输出目录
output_dir = "results/week1_day1"
os.makedirs(output_dir, exist_ok=True)

# 保存JSON
with open(f"{output_dir}/resnet18_summary.json", 'w') as f:
    json.dump(results, f, indent=2)

print("✅ 结果保存到: results/week1_day1/resnet18_summary.json")

# 生成ASR对比图
plt.figure(figsize=(10, 6))
methods = list(results.keys())
asrs = [results[m]['ASR'] for m in methods]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
plt.bar(methods, asrs, color=colors)
plt.ylabel('Attack Success Rate (%)', fontsize=12)
plt.title('ASR Comparison - ResNet18 (85% Accuracy)', fontsize=14, fontweight='bold')
plt.ylim(0, 100)
plt.grid(axis='y', alpha=0.3)

# 添加数值标签
for i, (method, asr) in enumerate(zip(methods, asrs)):
    plt.text(i, asr + 2, f'{asr:.1f}%', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{output_dir}/asr_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ ASR对比图保存到: results/week1_day1/asr_comparison.png")

# 生成L0对比图
plt.figure(figsize=(10, 6))
l0s = [results[m]['L0'] for m in methods]

plt.bar(methods, l0s, color=colors)
plt.ylabel('Average L0 Norm (Modified Pixels)', fontsize=12)
plt.title('L0 Norm Comparison - ResNet18 (85% Accuracy)', fontsize=14, fontweight='bold')
plt.ylim(0, max(l0s) * 1.2)
plt.grid(axis='y', alpha=0.3)

# 添加数值标签
for i, (method, l0) in enumerate(zip(methods, l0s)):
    plt.text(i, l0 + 0.1, f'{l0:.2f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{output_dir}/l0_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ L0对比图保存到: results/week1_day1/l0_comparison.png")

# 生成时间对比图
plt.figure(figsize=(10, 6))
times = [results[m]['Time'] for m in methods]

plt.bar(methods, times, color=colors)
plt.ylabel('Average Time per Sample (seconds)', fontsize=12)
plt.title('Time Efficiency Comparison - ResNet18', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# 添加数值标签
for i, (method, t) in enumerate(zip(methods, times)):
    plt.text(i, t + max(times)*0.03, f'{t:.2f}s', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{output_dir}/time_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ 时间对比图保存到: results/week1_day1/time_comparison.png")

# 打印总结
print("\n" + "=" * 80)
print("📊 Day 1 实验结果总结")
print("=" * 80)
print(f"\n模型: ResNet18 (准确率 85.1%)")
print(f"样本数: 100")
print(f"\n结果:")
for method, data in results.items():
    print(f"\n{method}:")
    print(f"  ASR: {data['ASR']:.1f}%")
    print(f"  L0: {data['L0']:.2f} pixels")
    print(f"  L2: {data['L2']:.4f}")
    print(f"  SSIM: {data['SSIM']:.4f}")
    print(f"  Time: {data['Time']:.3f}s")

print("\n" + "=" * 80)
print("✅ Day 1 任务完成！")
print("=" * 80)




















