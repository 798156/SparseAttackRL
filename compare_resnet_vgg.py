# compare_resnet_vgg.py
"""
对比ResNet18和VGG16的实验结果
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_results():
    """加载实验结果"""
    results = {}
    
    # ResNet18结果
    resnet_file = 'results/week1_day1/resnet18_summary.json'
    if os.path.exists(resnet_file):
        with open(resnet_file, 'r') as f:
            results['ResNet18'] = json.load(f)
        print(f"✅ 加载ResNet18结果: {resnet_file}")
    else:
        print(f"⚠️  ResNet18结果不存在: {resnet_file}")
    
    # VGG16结果
    vgg_file = 'results/week1_day2/vgg16_summary.json'
    if os.path.exists(vgg_file):
        with open(vgg_file, 'r') as f:
            results['VGG16'] = json.load(f)
        print(f"✅ 加载VGG16结果: {vgg_file}")
    else:
        print(f"⚠️  VGG16结果不存在: {vgg_file}")
    
    return results


def print_comparison(results):
    """打印对比表格"""
    print("\n" + "=" * 100)
    print("📊 ResNet18 vs VGG16 对比结果")
    print("=" * 100 + "\n")
    
    if 'ResNet18' not in results or 'VGG16' not in results:
        print("❌ 结果不完整，无法对比")
        return
    
    resnet_data = results['ResNet18']
    vgg_data = results['VGG16']
    
    # 获取所有攻击方法
    methods = set(resnet_data.keys()) | set(vgg_data.keys())
    
    # 打印表头
    print(f"{'攻击方法':<15} | {'指标':<8} | {'ResNet18':>12} | {'VGG16':>12} | {'差异':>12}")
    print("-" * 100)
    
    # 打印每个方法的结果
    for method in sorted(methods):
        if method in resnet_data and method in vgg_data:
            resnet_metrics = resnet_data[method]
            vgg_metrics = vgg_data[method]
            
            # ASR
            asr_diff = vgg_metrics['ASR'] - resnet_metrics['ASR']
            print(f"{method:<15} | {'ASR':<8} | {resnet_metrics['ASR']:>11.1f}% | {vgg_metrics['ASR']:>11.1f}% | {asr_diff:>+11.1f}%")
            
            # L0
            l0_diff = vgg_metrics['L0'] - resnet_metrics['L0']
            print(f"{'':15} | {'L0':<8} | {resnet_metrics['L0']:>12.2f} | {vgg_metrics['L0']:>12.2f} | {l0_diff:>+12.2f}")
            
            # L2
            l2_diff = vgg_metrics['L2'] - resnet_metrics['L2']
            print(f"{'':15} | {'L2':<8} | {resnet_metrics['L2']:>12.4f} | {vgg_metrics['L2']:>12.4f} | {l2_diff:>+12.4f}")
            
            # Time
            time_diff = vgg_metrics['Time'] - resnet_metrics['Time']
            print(f"{'':15} | {'Time':<8} | {resnet_metrics['Time']:>11.3f}s | {vgg_metrics['Time']:>11.3f}s | {time_diff:>+11.3f}s")
            print("-" * 100)
        elif method in resnet_data:
            print(f"{method:<15} | {'N/A':<8} | {'有数据':>12} | {'无数据':>12} | {'-':>12}")
            print("-" * 100)
        elif method in vgg_data:
            print(f"{method:<15} | {'N/A':<8} | {'无数据':>12} | {'有数据':>12} | {'-':>12}")
            print("-" * 100)


def plot_comparison(results):
    """绘制对比图"""
    if 'ResNet18' not in results or 'VGG16' not in results:
        print("\n❌ 结果不完整，无法绘制对比图")
        return
    
    resnet_data = results['ResNet18']
    vgg_data = results['VGG16']
    
    # 获取共同的攻击方法
    common_methods = set(resnet_data.keys()) & set(vgg_data.keys())
    common_methods = sorted(list(common_methods))
    
    if not common_methods:
        print("\n❌ 没有共同的攻击方法，无法绘制对比图")
        return
    
    # 准备数据
    resnet_asr = [resnet_data[m]['ASR'] for m in common_methods]
    vgg_asr = [vgg_data[m]['ASR'] for m in common_methods]
    
    resnet_l0 = [resnet_data[m]['L0'] for m in common_methods]
    vgg_l0 = [vgg_data[m]['L0'] for m in common_methods]
    
    resnet_time = [resnet_data[m]['Time'] for m in common_methods]
    vgg_time = [vgg_data[m]['Time'] for m in common_methods]
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x = np.arange(len(common_methods))
    width = 0.35
    
    # ASR对比
    axes[0].bar(x - width/2, resnet_asr, width, label='ResNet18', alpha=0.8)
    axes[0].bar(x + width/2, vgg_asr, width, label='VGG16', alpha=0.8)
    axes[0].set_xlabel('攻击方法', fontsize=12)
    axes[0].set_ylabel('ASR (%)', fontsize=12)
    axes[0].set_title('攻击成功率对比', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(common_methods, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # L0对比
    axes[1].bar(x - width/2, resnet_l0, width, label='ResNet18', alpha=0.8)
    axes[1].bar(x + width/2, vgg_l0, width, label='VGG16', alpha=0.8)
    axes[1].set_xlabel('攻击方法', fontsize=12)
    axes[1].set_ylabel('L0范数', fontsize=12)
    axes[1].set_title('稀疏性对比 (L0)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(common_methods, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # 时间对比
    axes[2].bar(x - width/2, resnet_time, width, label='ResNet18', alpha=0.8)
    axes[2].bar(x + width/2, vgg_time, width, label='VGG16', alpha=0.8)
    axes[2].set_xlabel('攻击方法', fontsize=12)
    axes[2].set_ylabel('时间 (秒)', fontsize=12)
    axes[2].set_title('效率对比', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(common_methods, rotation=45, ha='right')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = 'results/week1_day2'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'resnet_vs_vgg_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 对比图已保存: {output_file}")
    
    # 也保存PDF版本
    output_pdf = os.path.join(output_dir, 'resnet_vs_vgg_comparison.pdf')
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"📊 对比图已保存: {output_pdf}")
    
    plt.show()


def analyze_differences(results):
    """分析差异"""
    if 'ResNet18' not in results or 'VGG16' not in results:
        return
    
    print("\n" + "=" * 100)
    print("🔍 深入分析")
    print("=" * 100 + "\n")
    
    resnet_data = results['ResNet18']
    vgg_data = results['VGG16']
    
    common_methods = set(resnet_data.keys()) & set(vgg_data.keys())
    
    if not common_methods:
        print("❌ 没有共同的攻击方法")
        return
    
    # 计算平均差异
    asr_diffs = []
    l0_diffs = []
    
    for method in common_methods:
        asr_diff = vgg_data[method]['ASR'] - resnet_data[method]['ASR']
        l0_diff = vgg_data[method]['L0'] - resnet_data[method]['L0']
        asr_diffs.append(asr_diff)
        l0_diffs.append(l0_diff)
    
    avg_asr_diff = np.mean(asr_diffs)
    avg_l0_diff = np.mean(l0_diffs)
    
    print("📈 整体趋势：")
    print(f"  平均ASR差异: {avg_asr_diff:+.2f}% (VGG16 vs ResNet18)")
    print(f"  平均L0差异:  {avg_l0_diff:+.2f} (VGG16 vs ResNet18)")
    print()
    
    if avg_asr_diff > 0:
        print("✅ VGG16更容易被攻击（ASR更高）")
    elif avg_asr_diff < 0:
        print("✅ ResNet18更容易被攻击（ASR更高）")
    else:
        print("⚖️  两个模型差不多")
    print()
    
    if abs(avg_l0_diff) < 0.5:
        print("✅ 两个模型的稀疏性相似")
    else:
        print(f"⚠️  稀疏性有明显差异")
    print()
    
    # 找出最有效的攻击方法
    print("🎯 最有效的攻击方法：")
    
    resnet_best = max(common_methods, key=lambda m: resnet_data[m]['ASR'])
    vgg_best = max(common_methods, key=lambda m: vgg_data[m]['ASR'])
    
    print(f"  ResNet18: {resnet_best} (ASR={resnet_data[resnet_best]['ASR']:.1f}%)")
    print(f"  VGG16:    {vgg_best} (ASR={vgg_data[vgg_best]['ASR']:.1f}%)")
    print()
    
    # 找出最稀疏的攻击
    print("🎯 最稀疏的攻击（L0最小）：")
    
    # 只考虑成功的攻击
    resnet_sparse = min([m for m in common_methods if resnet_data[m]['ASR'] > 0], 
                        key=lambda m: resnet_data[m]['L0'], default=None)
    vgg_sparse = min([m for m in common_methods if vgg_data[m]['ASR'] > 0], 
                     key=lambda m: vgg_data[m]['L0'], default=None)
    
    if resnet_sparse:
        print(f"  ResNet18: {resnet_sparse} (L0={resnet_data[resnet_sparse]['L0']:.2f})")
    if vgg_sparse:
        print(f"  VGG16:    {vgg_sparse} (L0={vgg_data[vgg_sparse]['L0']:.2f})")
    print()


def main():
    print("=" * 100)
    print("🔍 ResNet18 vs VGG16 对比分析")
    print("=" * 100)
    
    # 加载结果
    results = load_results()
    
    if not results:
        print("\n❌ 没有找到任何结果文件")
        return
    
    # 打印对比
    print_comparison(results)
    
    # 分析差异
    analyze_differences(results)
    
    # 绘制对比图
    if len(results) >= 2:
        plot_comparison(results)
    
    print("\n" + "=" * 100)
    print("✅ 对比分析完成")
    print("=" * 100)


if __name__ == '__main__':
    main()








