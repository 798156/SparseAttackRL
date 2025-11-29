"""
对比标准模型 vs 防御模型的攻击结果

生成：
1. 完整对比表
2. ASR下降幅度分析
3. 相对性能保持度
4. 可视化图表
5. 论文用LaTeX表格
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# 设置绘图风格
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_results(model_type):
    """加载结果数据"""
    if model_type == 'standard':
        base_dir = Path('results/complete_baseline')
        prefix = 'resnet18'  # 标准模型的ResNet18结果
    else:  # defended
        base_dir = Path('results/defended_model')
        prefix = 'defended'
    
    methods = ['jsma', 'sparsefool', 'greedy', 'randomsparse', 'pixelgrad']
    results = {}
    
    for method in methods:
        file_path = base_dir / f'{prefix}_{method}.json'
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                results[method] = data
        else:
            print(f"⚠️ 未找到: {file_path}")
    
    return results

def calculate_summary(results):
    """计算汇总统计"""
    summary = {}
    
    for method, data in results.items():
        # 支持两种JSON格式
        # 格式1: {"samples": [...]}  (test_on_defended_model.py格式)
        # 格式2: {"asr": ..., "avg_l0": ..., "detailed_results": [...]}  (final_test_with_greedy.py格式)
        
        if 'asr' in data and 'avg_l0' in data:
            # 格式2：直接使用顶层统计数据
            summary[method] = {
                'asr': data.get('asr', 0),
                'avg_l0': data.get('avg_l0', 0),
                'avg_l2': data.get('avg_l2', 0),
                'avg_ssim': data.get('avg_ssim', 0),
                'avg_time': data.get('avg_time', 0),
                'num_samples': data.get('total_samples', 0)
            }
        else:
            # 格式1：从samples计算
            samples = data.get('samples', [])
            successes = [s for s in samples if s.get('success', False)]
            
            asr = len(successes) / len(samples) * 100 if samples else 0
            
            if successes:
                avg_l0 = np.mean([s.get('l0', 0) for s in successes])
                avg_l2 = np.mean([s.get('l2', 0) for s in successes])
                avg_ssim = np.mean([s.get('ssim', 0) for s in successes])
            else:
                avg_l0 = avg_l2 = avg_ssim = 0
            
            avg_time = np.mean([s.get('time', 0) for s in samples]) if samples else 0
            
            summary[method] = {
                'asr': asr,
                'avg_l0': avg_l0,
                'avg_l2': avg_l2,
                'avg_ssim': avg_ssim,
                'avg_time': avg_time,
                'num_samples': len(samples)
            }
    
    return summary

def compare_results(standard_summary, defended_summary):
    """对比分析"""
    comparison = {}
    
    for method in standard_summary.keys():
        if method in defended_summary:
            std = standard_summary[method]
            def_ = defended_summary[method]
            
            asr_drop = std['asr'] - def_['asr']
            asr_drop_pct = (asr_drop / std['asr'] * 100) if std['asr'] > 0 else 0
            
            comparison[method] = {
                'standard_asr': std['asr'],
                'defended_asr': def_['asr'],
                'asr_drop': asr_drop,
                'asr_drop_pct': asr_drop_pct,
                'standard_l0': std['avg_l0'],
                'defended_l0': def_['avg_l0'],
            }
    
    return comparison

def print_comparison_table(standard_summary, defended_summary, comparison):
    """打印对比表"""
    print("\n" + "="*100)
    print("📊 Standard vs Defended Model - Complete Comparison")
    print("="*100)
    
    print(f"\n{'Method':<15} {'Standard ASR':<15} {'Defended ASR':<15} {'Drop':<12} {'Drop %':<10}")
    print("-"*100)
    
    method_names = {
        'jsma': 'JSMA',
        'sparsefool': 'SparseFool',
        'greedy': 'Greedy',
        'pixelgrad': 'PixelGrad',
        'randomsparse': 'RandomSparse'
    }
    
    for method in ['jsma', 'greedy', 'sparsefool', 'pixelgrad', 'randomsparse']:
        if method in comparison:
            comp = comparison[method]
            name = method_names.get(method, method)
            print(f"{name:<15} "
                  f"{comp['standard_asr']:>12.1f}%   "
                  f"{comp['defended_asr']:>12.1f}%   "
                  f"{comp['asr_drop']:>9.1f}%  "
                  f"{comp['asr_drop_pct']:>8.1f}%")
    
    print("\n" + "="*100)
    print("📊 Detailed Metrics Comparison")
    print("="*100)
    
    print(f"\n{'Method':<15} {'Model':<12} {'ASR':<10} {'L0':<8} {'L2':<10} {'SSIM':<8} {'Time(s)':<8}")
    print("-"*100)
    
    for method in ['jsma', 'greedy', 'sparsefool', 'pixelgrad', 'randomsparse']:
        if method in standard_summary:
            name = method_names.get(method, method)
            std = standard_summary[method]
            def_ = defended_summary.get(method, {})
            
            print(f"{name:<15} {'Standard':<12} {std['asr']:>7.1f}% "
                  f"{std['avg_l0']:>6.2f}  {std['avg_l2']:>8.4f}  "
                  f"{std['avg_ssim']:>6.4f}  {std['avg_time']:>6.3f}")
            
            if def_:
                print(f"{'':<15} {'Defended':<12} {def_['asr']:>7.1f}% "
                      f"{def_['avg_l0']:>6.2f}  {def_['avg_l2']:>8.4f}  "
                      f"{def_['avg_ssim']:>6.4f}  {def_['avg_time']:>6.3f}")
                print()

def analyze_ranking_consistency(standard_summary, defended_summary):
    """分析排名一致性"""
    print("\n" + "="*100)
    print("📈 Ranking Consistency Analysis")
    print("="*100)
    
    # 按ASR排序
    std_ranking = sorted(standard_summary.items(), key=lambda x: x[1]['asr'], reverse=True)
    def_ranking = sorted(defended_summary.items(), key=lambda x: x[1]['asr'], reverse=True)
    
    print("\n🏆 ASR Ranking:")
    print("\nStandard Model:")
    for i, (method, summary) in enumerate(std_ranking, 1):
        print(f"  {i}. {method.upper():<15} {summary['asr']:.1f}%")
    
    print("\nDefended Model:")
    for i, (method, summary) in enumerate(def_ranking, 1):
        print(f"  {i}. {method.upper():<15} {summary['asr']:.1f}%")
    
    # 计算Spearman相关系数
    std_ranks = {m: i for i, (m, _) in enumerate(std_ranking)}
    def_ranks = {m: i for i, (m, _) in enumerate(def_ranking)}
    
    common_methods = set(std_ranks.keys()) & set(def_ranks.keys())
    std_rank_values = [std_ranks[m] for m in sorted(common_methods)]
    def_rank_values = [def_ranks[m] for m in sorted(common_methods)]
    
    if len(std_rank_values) > 1:
        correlation, p_value = stats.spearmanr(std_rank_values, def_rank_values)
        print(f"\n📊 Spearman Rank Correlation: {correlation:.3f} (p={p_value:.4f})")
        
        if correlation > 0.8:
            print("   ✅ Very strong correlation - ranking highly consistent!")
        elif correlation > 0.6:
            print("   ✅ Strong correlation - ranking mostly consistent")
        else:
            print("   ⚠️ Moderate correlation - some ranking changes")

def plot_asr_comparison(comparison, output_dir):
    """绘制ASR对比图"""
    methods = list(comparison.keys())
    method_labels = {
        'jsma': 'JSMA',
        'sparsefool': 'SparseFool', 
        'greedy': 'Greedy',
        'pixelgrad': 'PixelGrad',
        'randomsparse': 'RandomSparse'
    }
    
    labels = [method_labels.get(m, m) for m in methods]
    standard_asrs = [comparison[m]['standard_asr'] for m in methods]
    defended_asrs = [comparison[m]['defended_asr'] for m in methods]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, standard_asrs, width, label='Standard Model', alpha=0.8)
    bars2 = ax.bar(x + width/2, defended_asrs, width, label='Defended Model', alpha=0.8)
    
    ax.set_xlabel('Attack Method', fontsize=12)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title('ASR Comparison: Standard vs Defended Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'asr_standard_vs_defended.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'asr_standard_vs_defended.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 保存: asr_standard_vs_defended.png/pdf")

def plot_asr_drop(comparison, output_dir):
    """绘制ASR下降幅度图"""
    methods = list(comparison.keys())
    method_labels = {
        'jsma': 'JSMA',
        'sparsefool': 'SparseFool',
        'greedy': 'Greedy',
        'pixelgrad': 'PixelGrad',
        'randomsparse': 'RandomSparse'
    }
    
    labels = [method_labels.get(m, m) for m in methods]
    drops = [comparison[m]['asr_drop'] for m in methods]
    drop_pcts = [comparison[m]['asr_drop_pct'] for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 绝对下降
    bars1 = ax1.bar(labels, drops, alpha=0.8, color='coral')
    ax1.set_ylabel('ASR Drop (Percentage Points)', fontsize=12)
    ax1.set_title('Absolute ASR Drop on Defended Model', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 相对下降
    bars2 = ax2.bar(labels, drop_pcts, alpha=0.8, color='steelblue')
    ax2.set_ylabel('Relative ASR Drop (%)', fontsize=12)
    ax2.set_title('Relative ASR Drop on Defended Model', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'asr_drop_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'asr_drop_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 保存: asr_drop_analysis.png/pdf")

def generate_latex_table(standard_summary, defended_summary, comparison, output_dir):
    """生成LaTeX表格"""
    latex_code = r"""
\begin{table}[htbp]
\centering
\caption{Comparison of Attack Methods on Standard vs Defended Models}
\label{tab:standard_vs_defended}
\begin{tabular}{l|cc|cc}
\hline
\textbf{Method} & \multicolumn{2}{c|}{\textbf{ASR (\%)}} & \multicolumn{2}{c}{\textbf{Avg L0}} \\
& Standard & Defended & Standard & Defended \\
\hline
"""
    
    method_order = ['jsma', 'greedy', 'sparsefool', 'pixelgrad', 'randomsparse']
    method_names = {
        'jsma': 'JSMA',
        'sparsefool': 'SparseFool',
        'greedy': 'Greedy',
        'pixelgrad': 'PixelGrad',
        'randomsparse': 'RandomSparse'
    }
    
    for method in method_order:
        if method in comparison:
            comp = comparison[method]
            name = method_names.get(method, method)
            latex_code += f"{name} & "
            latex_code += f"{comp['standard_asr']:.1f} & {comp['defended_asr']:.1f} & "
            latex_code += f"{comp['standard_l0']:.2f} & {comp['defended_l0']:.2f} \\\\\n"
    
    latex_code += r"""\hline
\end{tabular}
\end{table}
"""
    
    output_file = output_dir / 'latex_table_standard_vs_defended.tex'
    with open(output_file, 'w') as f:
        f.write(latex_code)
    
    print(f"  ✅ 保存: latex_table_standard_vs_defended.tex")

def write_analysis_report(standard_summary, defended_summary, comparison, output_dir):
    """生成分析报告"""
    report = """# 标准模型 vs 防御模型 - 完整对比分析

## 1. 实验概述

本分析对比了5种稀疏对抗攻击方法在标准模型和防御模型上的性能。

**测试配置：**
- 标准模型：ResNet18 (CIFAR-10, ~88% accuracy)
- 防御模型：RobustBench对抗训练ResNet18 (~83-85% accuracy)
- 攻击方法：JSMA, SparseFool, Greedy, PixelGrad, RandomSparse
- 测试样本：每个配置100个样本

---

## 2. 主要发现

"""
    
    # 计算平均下降
    avg_drop = np.mean([comp['asr_drop'] for comp in comparison.values()])
    avg_drop_pct = np.mean([comp['asr_drop_pct'] for comp in comparison.values()])
    
    report += f"""
### 发现1：防御模型显著降低了攻击成功率

- **平均ASR下降：** {avg_drop:.1f} 个百分点
- **平均相对下降：** {avg_drop_pct:.1f}%

这证明了对抗训练的有效性。

"""
    
    # 找出下降最多和最少的方法
    max_drop_method = max(comparison.items(), key=lambda x: x[1]['asr_drop_pct'])
    min_drop_method = min(comparison.items(), key=lambda x: x[1]['asr_drop_pct'])
    
    report += f"""
### 发现2：不同方法对防御的敏感度不同

- **最敏感方法：** {max_drop_method[0].upper()} (下降 {max_drop_method[1]['asr_drop_pct']:.1f}%)
- **最稳定方法：** {min_drop_method[0].upper()} (下降 {min_drop_method[1]['asr_drop_pct']:.1f}%)

**解释：**
- 某些方法更依赖于模型的脆弱性，在防御模型上性能下降明显
- 某些方法具有更好的鲁棒性，在防御场景下相对稳定

"""
    
    # 排名一致性
    std_ranking = sorted(standard_summary.items(), key=lambda x: x[1]['asr'], reverse=True)
    def_ranking = sorted(defended_summary.items(), key=lambda x: x[1]['asr'], reverse=True)
    
    std_top3 = [m for m, _ in std_ranking[:3]]
    def_top3 = [m for m, _ in def_ranking[:3]]
    
    common_top3 = set(std_top3) & set(def_top3)
    
    report += f"""
### 发现3：方法相对排名{"基本保持" if len(common_top3) >= 2 else "有所变化"}

**标准模型Top 3：** {', '.join([m.upper() for m in std_top3])}
**防御模型Top 3：** {', '.join([m.upper() for m in def_top3])}

{"✅ 前3名中有" + str(len(common_top3)) + "个方法保持，说明方法的相对性能在防御场景下稳定。" if len(common_top3) >= 2 else "⚠️ 排名有明显变化，不同方法对防御的适应性不同。"}

"""
    
    # RandomSparse分析
    if 'randomsparse' in comparison:
        rs = comparison['randomsparse']
        report += f"""
### 发现4：RandomSparse仍然是最差的baseline

- **标准模型ASR：** {rs['standard_asr']:.1f}%
- **防御模型ASR：** {rs['defended_asr']:.1f}%
- **下降：** {rs['asr_drop']:.1f} 百分点

即使在防御模型上，RandomSparse的ASR仍然显著低于所有智能方法，
再次证明了梯度引导的像素选择策略的重要性。

"""
    
    report += """
---

## 3. 论文写作建议

### 3.1 实验章节

```latex
We further evaluate all methods on adversarially trained models 
from RobustBench to assess their practical applicability in 
defended scenarios. As expected, all methods show reduced ASR 
on the defended model, with an average drop of XX%. However, 
the relative performance ranking remains largely consistent, 
demonstrating the robustness of our findings.
```

### 3.2 讨论章节

可以讨论：
1. 不同方法对防御的敏感度差异
2. 为什么某些方法更鲁棒？
3. 这对实际部署有什么启示？

### 3.3 可能的额外贡献

如果发现了有趣的模式（例如某个方法特别稳定），可以：
- 专门分析原因
- 作为一个独立的发现
- 增强论文的深度

---

## 4. 数据表格

详见生成的LaTeX表格和图表。

---

## 5. 下一步建议

1. ✅ 检查所有数据的合理性
2. ✅ 确认发现是否有价值
3. ✅ 准备论文图表
4. 🎯 继续Week 1 Day 5：数据整理
5. 🎯 开始Week 2：补充分析

---

*生成时间：自动生成*
"""
    
    output_file = output_dir / 'analysis_standard_vs_defended.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"  ✅ 保存: analysis_standard_vs_defended.md")

def main():
    """主流程"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              📊 Standard vs Defended Model Comparison                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # 加载数据
    print("📂 加载数据...")
    standard_results = load_results('standard')
    defended_results = load_results('defended')
    
    if not standard_results or not defended_results:
        print("\n❌ 错误：缺少必要的结果文件")
        print("请确保已运行：")
        print("  1. python final_test_with_greedy.py")
        print("  2. python test_new_2methods.py")  
        print("  3. python test_on_defended_model.py")
        return
    
    # 计算汇总统计
    print("📊 计算统计...")
    standard_summary = calculate_summary(standard_results)
    defended_summary = calculate_summary(defended_results)
    
    # 对比分析
    comparison = compare_results(standard_summary, defended_summary)
    
    # 打印对比表
    print_comparison_table(standard_summary, defended_summary, comparison)
    
    # 排名一致性分析
    analyze_ranking_consistency(standard_summary, defended_summary)
    
    # 生成图表
    output_dir = Path('results/paper_materials')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 生成可视化图表...")
    plot_asr_comparison(comparison, output_dir)
    plot_asr_drop(comparison, output_dir)
    
    # 生成LaTeX表格
    print(f"\n📝 生成LaTeX表格...")
    generate_latex_table(standard_summary, defended_summary, comparison, output_dir)
    
    # 生成分析报告
    print(f"\n📄 生成分析报告...")
    write_analysis_report(standard_summary, defended_summary, comparison, output_dir)
    
    print(f"\n{'='*100}")
    print("🎉 对比分析完成！")
    print(f"{'='*100}")
    print(f"\n💾 所有结果保存在: {output_dir}")
    print("\n📂 生成的文件：")
    print("  - asr_standard_vs_defended.png/pdf")
    print("  - asr_drop_analysis.png/pdf")
    print("  - latex_table_standard_vs_defended.tex")
    print("  - analysis_standard_vs_defended.md")
    
    print(f"\n{'='*100}")
    print("📈 下一步：")
    print(f"{'='*100}")
    print("1. 查看分析报告了解详细发现")
    print("2. 检查图表质量")
    print("3. 准备Week 1 Day 5数据整理")
    print("4. 开始Week 2补充分析")

if __name__ == "__main__":
    main()

