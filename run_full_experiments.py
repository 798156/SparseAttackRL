# run_full_experiments.py
"""
完整实验矩阵
实验配置：
- 数据集：CIFAR-10, CIFAR-100, ImageNet（可选）
- 模型：ResNet18, VGG16, MobileNetV2, DenseNet121
- 攻击方法：SparseAttackRL V2, JSMA, One-Pixel, SparseFool
- 样本数：500/数据集

运行方式：
python run_full_experiments.py --quick_test  # 快速测试（100样本，2模型）
python run_full_experiments.py --full        # 完整实验（500样本，4模型）
"""

import torch
import numpy as np
import os
import time
import argparse
import json
from tqdm import tqdm
from datetime import datetime

# 设置matplotlib后端（避免Qt错误）
import matplotlib
matplotlib.use('Agg')  # 使用无GUI后端
import matplotlib.pyplot as plt
import seaborn as sns

# 导入自定义模块
from dataset_loader import DatasetLoader, get_all_datasets
from model_loader import ModelLoader, get_experiment_models
from sparse_attack_env_v2 import SparseAttackEnvV2
from ppo_trainer_v2 import train_rl_agent_v2
from one_pixel_attack import one_pixel_attack
from jsma_attack import jsma_attack
from sparsefool_attack import sparsefool_attack_simple
from evaluation_metrics import MetricsAggregator, compute_all_metrics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy import stats as scipy_stats
import pandas as pd


class FullExperimentRunner:
    """完整实验矩阵运行器"""
    
    def __init__(self, config):
        """
        参数:
            config: 实验配置字典
        """
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu'
        
        # 创建结果目录
        self.exp_dir = config['exp_dir']
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(f"{self.exp_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.exp_dir}/models", exist_ok=True)
        os.makedirs(f"{self.exp_dir}/logs", exist_ok=True)
        
        print("\n" + "=" * 80)
        print("🔬 完整实验矩阵初始化")
        print("=" * 80)
        print(f"📁 实验目录: {self.exp_dir}")
        print(f"🚀 使用设备: {self.device}")
        print(f"📊 样本数/数据集: {config['num_samples']}")
        print(f"🔄 最大修改步数: {config['max_steps']}")
        
        # 加载数据集
        self.datasets = self._load_datasets()
        
        # 所有结果
        self.all_results = {}
    
    def _load_datasets(self):
        """加载所有数据集"""
        print("\n📦 加载数据集...")
        
        datasets = {}
        dataset_names = self.config.get('datasets', ['cifar10'])
        
        for name in dataset_names:
            try:
                loader = DatasetLoader(name, self.config['data_root'])
                test_set = loader.load_test_set()
                
                if test_set is not None:
                    # 采样子集
                    indices = loader.get_sample_subset(
                        test_set, 
                        num_samples=self.config['num_samples']
                    )
                    
                    datasets[name] = {
                        'loader': loader,
                        'test_set': test_set,
                        'indices': indices
                    }
                    print(f"  ✅ {name.upper()}: {len(indices)} 样本")
            except Exception as e:
                print(f"  ❌ {name} 加载失败: {e}")
        
        return datasets
    
    def train_agents(self):
        """为每个数据集训练RL智能体"""
        print("\n" + "=" * 80)
        print("🎓 训练 RL 智能体")
        print("=" * 80)
        
        agents = {}
        
        for dataset_name, dataset_info in self.datasets.items():
            print(f"\n🔧 训练 {dataset_name.upper()} 的智能体...")
            
            agent_path = f"{self.exp_dir}/models/agent_{dataset_name}.zip"
            
            if os.path.exists(agent_path) and not self.config.get('retrain', False):
                print(f"  ✅ 模型已存在，直接加载")
                try:
                    agents[dataset_name] = PPO.load(agent_path)
                    continue
                except:
                    print(f"  ⚠️ 加载失败，重新训练")
            
            # 获取第一个样本用于训练
            test_set = dataset_info['test_set']
            indices = dataset_info['indices']
            image, label = test_set[indices[0]]
            
            # 创建环境
            loader = dataset_info['loader']
            model = ModelLoader.load_model(
                'resnet18',  # 训练时用ResNet18
                num_classes=loader.num_classes
            ).to(self.device)
            
            env = SparseAttackEnvV2(
                image, label, model,
                max_steps=self.config['max_steps'],
                use_saliency=True
            )
            
            # 训练
            agent = train_rl_agent_v2(
                env,
                timesteps=self.config['train_timesteps'],
                save_path=agent_path.replace('.zip', ''),
                use_cnn=True
            )
            
            agents[dataset_name] = agent
        
        return agents
    
    def run_single_experiment(self, dataset_name, model_name, method_name, 
                             agent, model, test_set, indices, max_steps):
        """
        运行单个实验组合
        
        返回:
            results: MetricsAggregator
        """
        results = MetricsAggregator()
        loader = self.datasets[dataset_name]['loader']
        
        for idx in tqdm(indices, desc=f"{dataset_name}|{model_name}|{method_name}", leave=False):
            image, label = test_set[idx]
            
            start_time = time.time()
            success = False
            l0_norm = 0
            
            try:
                if method_name == 'rl_v2':
                    # SparseAttackRL V2
                    env = SparseAttackEnvV2(image, label, model, max_steps=max_steps, use_saliency=True)
                    vec_env = DummyVecEnv([lambda: env])
                    obs = vec_env.reset()
                    
                    done = False
                    steps = 0
                    
                    while not done and steps < max_steps:
                        action, _ = agent.predict(obs)
                        result = vec_env.step(action)
                        
                        if len(result) == 4:
                            obs, _, done, info = result
                        else:
                            obs, _, terminated, truncated, info = result
                            done = terminated[0] or truncated[0]
                        
                        info = info[0] if isinstance(info, list) else info
                        steps += 1
                        
                        if info.get('success', False):
                            success = True
                            adv_img = env.current_image.squeeze(0)
                            break
                    
                    if success:
                        metrics = compute_all_metrics(image, adv_img)
                        l0_norm = metrics['l0_norm']
                
                elif method_name == 'jsma':
                    # JSMA Attack
                    # 增大theta以确保修改足够大
                    success, adv_img, pixels = jsma_attack(
                        image, label, model, max_pixels=max_steps, theta=10.0
                    )
                    if success:
                        metrics = compute_all_metrics(image, adv_img)
                        l0_norm = metrics['l0_norm']
                
                elif method_name == 'one_pixel':
                    # One-Pixel Attack
                    success, params = one_pixel_attack(
                        image, label, model, max_iter=100
                    )
                    if success:
                        l0_norm = 1.0
                
                elif method_name == 'sparsefool':
                    # SparseFool Attack
                    success, adv_img, pixels = sparsefool_attack_simple(
                        image, label, model, max_pixels=max_steps
                    )
                    if success:
                        metrics = compute_all_metrics(image, adv_img)
                        l0_norm = metrics['l0_norm']
            
            except Exception as e:
                # print(f"  ⚠️ Error [{idx}]: {e}")
                pass
            
            attack_time = time.time() - start_time
            
            # 记录结果
            results.add(
                success=success,
                attack_time=attack_time,
                query_count=max_steps if not success else (l0_norm if l0_norm > 0 else 1),
                l0_norm=l0_norm if success else 0,
                l2_norm=0,  # 简化
                linf_norm=0,
                ssim=0,
                psnr=0
            )
        
        return results
    
    def run_all_experiments(self, agents):
        """运行所有实验组合"""
        print("\n" + "=" * 80)
        print("🚀 开始运行完整实验矩阵")
        print("=" * 80)
        
        # 实验矩阵
        methods = ['rl_v2', 'jsma', 'one_pixel', 'sparsefool']
        
        # 总进度
        total_experiments = 0
        for dataset_name in self.datasets.keys():
            models = get_experiment_models(
                num_classes=self.datasets[dataset_name]['loader'].num_classes,
                quick_test=self.config.get('quick_test', False)
            )
            total_experiments += len(models) * len(methods)
        
        print(f"\n📊 总实验数: {total_experiments}")
        print(f"   数据集: {len(self.datasets)}")
        print(f"   模型/数据集: {len(models)}")
        print(f"   方法: {len(methods)}")
        print("=" * 80 + "\n")
        
        # 运行实验
        exp_count = 0
        
        for dataset_name, dataset_info in self.datasets.items():
            print(f"\n{'='*80}")
            print(f"📦 数据集: {dataset_name.upper()}")
            print(f"{'='*80}")
            
            test_set = dataset_info['test_set']
            indices = dataset_info['indices']
            agent = agents.get(dataset_name)
            
            # 加载该数据集的所有模型
            num_classes = dataset_info['loader'].num_classes
            models = get_experiment_models(
                num_classes=num_classes,
                quick_test=self.config.get('quick_test', False)
            )
            
            for model_name, model in models.items():
                print(f"\n🔧 模型: {model_name.upper()}")
                model = model.to(self.device).eval()
                
                for method_name in methods:
                    exp_count += 1
                    print(f"  [{exp_count}/{total_experiments}] {method_name.upper()}...")
                    
                    # 运行实验
                    results = self.run_single_experiment(
                        dataset_name, model_name, method_name,
                        agent, model, test_set, indices,
                        self.config['max_steps']
                    )
                    
                    # 保存结果
                    key = f"{dataset_name}_{model_name}_{method_name}"
                    self.all_results[key] = {
                        'dataset': dataset_name,
                        'model': model_name,
                        'method': method_name,
                        'results': results,
                        'stats': results.compute_statistics()
                    }
                    
                    # 显示简要结果
                    stats = self.all_results[key]['stats']
                    asr = stats.get('success_rate', 0) * 100
                    l0 = stats.get('l0_norm_mean', 0)
                    print(f"      ASR: {asr:.1f}%, L0: {l0:.2f}")
        
        print("\n" + "=" * 80)
        print("✅ 所有实验完成！")
        print("=" * 80)
    
    def save_results(self):
        """保存所有结果"""
        print("\n💾 保存结果...")
        
        # 1. 保存JSON格式的统计信息
        stats_dict = {}
        for key, value in self.all_results.items():
            stats_dict[key] = value['stats']
        
        with open(f"{self.exp_dir}/all_stats.json", 'w') as f:
            json.dump(stats_dict, f, indent=2, default=str)
        print(f"  ✅ 统计信息: {self.exp_dir}/all_stats.json")
        
        # 2. 保存CSV格式的详细数据
        for key, value in self.all_results.items():
            csv_path = f"{self.exp_dir}/logs/{key}_metrics.csv"
            value['results'].save_to_csv(csv_path)
        print(f"  ✅ 详细数据: {self.exp_dir}/logs/")
        
        # 3. 生成综合表格
        self._generate_summary_table()
    
    def _generate_summary_table(self):
        """生成综合结果表格"""
        print("\n📊 生成综合表格...")
        
        # 创建DataFrame
        rows = []
        for key, value in self.all_results.items():
            stats = value['stats']
            rows.append({
                'Dataset': value['dataset'],
                'Model': value['model'],
                'Method': value['method'],
                'ASR (%)': stats.get('success_rate', 0) * 100,
                'Avg L0': stats.get('l0_norm_mean', 0),
                'Avg Time (s)': stats.get('attack_time_mean', 0),
            })
        
        df = pd.DataFrame(rows)
        
        # 保存为CSV
        csv_path = f"{self.exp_dir}/summary_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"  ✅ 综合表格: {csv_path}")
        
        # 打印表格
        print("\n" + "=" * 100)
        print("📊 综合实验结果")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)
    
    def generate_visualizations(self):
        """生成可视化图表"""
        print("\n📈 生成可视化图表...")
        
        sns.set(style="whitegrid", font_scale=1.2)
        
        # 为每个数据集生成图表
        for dataset_name in self.datasets.keys():
            self._generate_dataset_plots(dataset_name)
        
        # 生成跨数据集对比
        self._generate_cross_dataset_plots()
        
        print(f"  ✅ 图表保存至: {self.exp_dir}/plots/")
    
    def _generate_dataset_plots(self, dataset_name):
        """为单个数据集生成图表"""
        # 提取该数据集的所有结果
        dataset_results = {k: v for k, v in self.all_results.items() 
                          if v['dataset'] == dataset_name}
        
        if not dataset_results:
            return
        
        # 按方法聚合（平均所有模型）
        methods = {}
        for key, value in dataset_results.items():
            method = value['method']
            if method not in methods:
                methods[method] = []
            methods[method].append(value['stats'].get('success_rate', 0) * 100)
        
        # 计算平均ASR
        method_names = []
        asrs = []
        for method, asr_list in methods.items():
            method_names.append(method.upper())
            asrs.append(np.mean(asr_list))
        
        # 绘制柱状图
        plt.figure(figsize=(10, 6))
        colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
        bars = plt.bar(method_names, asrs, color=colors[:len(method_names)], alpha=0.8)
        plt.ylabel('Attack Success Rate (%)', fontsize=14)
        plt.title(f'ASR Comparison on {dataset_name.upper()}', fontsize=16, weight='bold')
        plt.ylim(0, 105)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=12)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.exp_dir}/plots/asr_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.exp_dir}/plots/asr_{dataset_name}.pdf", bbox_inches='tight')
        plt.close()
    
    def _generate_cross_dataset_plots(self):
        """生成跨数据集对比图"""
        # 这里可以生成更复杂的对比图
        pass


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='完整实验矩阵')
    parser.add_argument('--quick_test', action='store_true', 
                       help='快速测试（100样本，2模型）')
    parser.add_argument('--full', action='store_true',
                       help='完整实验（500样本，4模型）')
    parser.add_argument('--datasets', nargs='+', default=['cifar10'],
                       help='数据集列表 (cifar10, cifar100, imagenet)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='样本数（覆盖默认值）')
    parser.add_argument('--max_steps', type=int, default=5,
                       help='最大修改步数')
    parser.add_argument('--exp_dir', type=str, default='results/full_experiments',
                       help='实验结果目录')
    parser.add_argument('--no_gpu', action='store_true',
                       help='不使用GPU')
    parser.add_argument('--retrain', action='store_true',
                       help='重新训练RL智能体')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 构建配置
    config = {
        'data_root': './data',
        'exp_dir': args.exp_dir,
        'max_steps': args.max_steps,
        'use_gpu': not args.no_gpu,
        'retrain': args.retrain,
        'quick_test': args.quick_test,
        'datasets': args.datasets if not args.quick_test else ['cifar10'],
        'train_timesteps': 5000 if args.quick_test else 10000,
    }
    
    # 样本数
    if args.num_samples:
        config['num_samples'] = args.num_samples
    elif args.quick_test:
        config['num_samples'] = 100
    elif args.full:
        config['num_samples'] = 500
    else:
        config['num_samples'] = 100  # 默认
    
    # 创建实验运行器
    runner = FullExperimentRunner(config)
    
    # 1. 训练RL智能体
    agents = runner.train_agents()
    
    # 2. 运行所有实验
    runner.run_all_experiments(agents)
    
    # 3. 保存结果
    runner.save_results()
    
    # 4. 生成可视化
    runner.generate_visualizations()
    
    print("\n" + "=" * 80)
    print("🎉 所有实验完成！")
    print(f"📁 结果保存在: {config['exp_dir']}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

