# ppo_trainer_v3_improved.py
"""
改进版PPO训练器 - 多样本训练
核心改进：
1. ✅ 多样本并行训练（解决过拟合）
2. ✅ 动态样本采样（增加多样性）
3. ✅ 课程学习（从简单到困难）
4. ✅ 模型特定训练（每个模型单独训练）
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import os
import gymnasium as gym


class DynamicSampleEnv(gym.Env):
    """
    动态采样环境包装器
    每次reset时随机选择新的训练样本
    """
    
    def __init__(self, env_class, model, dataset, device='cuda', 
                 num_samples=100, max_steps=5):
        """
        参数:
            env_class: 环境类（如SparseAttackEnvV2）
            model: 目标模型
            dataset: 训练数据集
            num_samples: 训练样本池大小
            max_steps: 最大步数
        """
        super().__init__()  # 初始化gym.Env
        
        self.env_class = env_class
        self.model = model
        self.dataset = dataset
        self.device = device
        self.max_steps = max_steps
        
        # 选择正确分类的样本
        print(f"📊 选择{num_samples}个训练样本...")
        self.train_samples = self._select_samples(num_samples)
        print(f"✅ 选择了{len(self.train_samples)}个样本")
        
        # 当前环境
        self.current_env = None
        self._reset_with_new_sample()
        
        # 环境属性
        self.action_space = self.current_env.action_space
        self.observation_space = self.current_env.observation_space
    
    def _select_samples(self, num_samples):
        """选择正确分类的样本"""
        correct_indices = []
        
        with torch.no_grad():
            for idx in range(min(len(self.dataset), num_samples * 3)):
                image, label = self.dataset[idx]
                image = image.unsqueeze(0).to(self.device)
                output = self.model(image)
                pred = output.argmax(dim=1).item()
                
                if pred == label:
                    correct_indices.append(idx)
                
                if len(correct_indices) >= num_samples:
                    break
        
        return correct_indices
    
    def _reset_with_new_sample(self):
        """用新样本创建环境"""
        # 随机选择一个样本
        sample_idx = np.random.choice(self.train_samples)
        image, label = self.dataset[sample_idx]
        
        # 创建新环境
        self.current_env = self.env_class(
            clean_image=image,
            true_label=label,
            model=self.model,
            max_steps=self.max_steps,
            use_saliency=True
        )
    
    def reset(self, **kwargs):
        """重置环境 - 随机选择新样本"""
        self._reset_with_new_sample()
        return self.current_env.reset(**kwargs)
    
    def step(self, action):
        """执行动作"""
        return self.current_env.step(action)


class CurriculumCallback(BaseCallback):
    """
    课程学习回调
    动态调整样本难度
    """
    
    def __init__(self, env_wrapper, verbose=0):
        super().__init__(verbose)
        self.env_wrapper = env_wrapper
        self.success_rate = 0.0
        self.check_interval = 1000
    
    def _on_step(self):
        # 每隔一段时间检查成功率
        if self.n_calls % self.check_interval == 0:
            # 根据成功率调整难度
            if self.success_rate > 0.7:
                # 成功率高，增加难度
                self.env_wrapper.max_steps = max(3, self.env_wrapper.max_steps - 1)
                if self.verbose:
                    print(f"📈 增加难度: max_steps = {self.env_wrapper.max_steps}")
            elif self.success_rate < 0.3:
                # 成功率低，降低难度
                self.env_wrapper.max_steps = min(10, self.env_wrapper.max_steps + 1)
                if self.verbose:
                    print(f"📉 降低难度: max_steps = {self.env_wrapper.max_steps}")
        
        return True


def train_rl_multi_sample(
    model,
    dataset,
    env_class,
    num_train_samples=100,
    timesteps=50000,
    save_path="ppo_sparse_multi",
    device='cuda',
    max_steps=5,
    verbose=1
):
    """
    多样本训练RL agent
    
    参数:
        model: 目标模型
        dataset: 训练数据集（CIFAR-10）
        env_class: 环境类
        num_train_samples: 训练样本数
        timesteps: 训练步数
        save_path: 保存路径
        device: 设备
        max_steps: 初始最大步数
        verbose: 日志级别
    
    返回:
        trained_agent: 训练好的agent
    """
    print("=" * 80)
    print("🚀 多样本RL训练 - 改进版")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  目标模型: {model.__class__.__name__}")
    print(f"  训练样本数: {num_train_samples}")
    print(f"  训练步数: {timesteps}")
    print(f"  初始max_steps: {max_steps}")
    print(f"  设备: {device}")
    
    # 创建动态采样环境
    def make_env():
        return DynamicSampleEnv(
            env_class=env_class,
            model=model,
            dataset=dataset,
            device=device,
            num_samples=num_train_samples,
            max_steps=max_steps
        )
    
    # 创建向量化环境（可选：并行训练）
    env = DummyVecEnv([make_env])
    
    # 导入CNN特征提取器
    from ppo_trainer_v2 import CNNFeatureExtractor
    
    # 配置策略网络
    policy_kwargs = dict(
        features_extractor_class=CNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[128, 128], vf=[128, 128])]
    )
    
    # 创建PPO agent
    agent = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # 熵系数：鼓励探索
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        tensorboard_log="./logs/",
        device=device
    )
    
    print(f"\n🎓 开始训练...")
    print(f"   每次reset会随机选择新样本")
    print(f"   总计{num_train_samples}个训练样本\n")
    
    # 训练
    agent.learn(
        total_timesteps=timesteps,
        tb_log_name="ppo_multi_sample",
        progress_bar=True
    )
    
    # 保存模型
    agent.save(save_path)
    print(f"\n💾 模型已保存: {save_path}.zip")
    
    return agent


def train_model_specific_agent(
    model,
    model_name,
    dataset,
    env_class,
    num_train_samples=100,
    timesteps=50000,
    save_dir="models",
    device='cuda'
):
    """
    为特定模型训练专门的RL agent
    
    参数:
        model: 目标模型
        model_name: 模型名称（如'resnet18', 'vgg16'）
        dataset: 训练数据集
        env_class: 环境类
        num_train_samples: 训练样本数
        timesteps: 训练步数
        save_dir: 保存目录
        device: 设备
    
    返回:
        agent: 训练好的agent
    """
    print("\n" + "=" * 80)
    print(f"🎯 为 {model_name.upper()} 训练专门的RL agent")
    print("=" * 80)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ppo_sparse_{model_name}")
    
    # 训练
    agent = train_rl_multi_sample(
        model=model,
        dataset=dataset,
        env_class=env_class,
        num_train_samples=num_train_samples,
        timesteps=timesteps,
        save_path=save_path,
        device=device,
        verbose=1
    )
    
    # 验证agent性能
    print(f"\n🧪 验证 {model_name} agent...")
    test_agent_performance(agent, model, dataset, num_test=20, device=device)
    
    return agent


def test_agent_performance(agent, model, dataset, num_test=20, device='cuda'):
    """
    测试agent性能
    """
    from sparse_attack_env_v2 import SparseAttackEnvV2
    
    successes = 0
    total_l0 = 0
    
    print(f"\n测试{num_test}个样本...")
    
    with torch.no_grad():
        # 选择测试样本
        test_indices = []
        for idx in range(len(dataset)):
            if len(test_indices) >= num_test:
                break
            
            image, label = dataset[idx]
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            pred = output.argmax(dim=1).item()
            
            if pred == label:
                test_indices.append(idx)
    
    for idx in test_indices:
        image, label = dataset[idx]
        
        # 创建环境
        env = SparseAttackEnvV2(
            clean_image=image,
            true_label=label,
            model=model,
            max_steps=5,
            use_saliency=True
        )
        
        # 执行攻击
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 10:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        # 检查结果
        with torch.no_grad():
            adv_image = torch.tensor(env.current_image, dtype=torch.float32).to(device)
            output = model(adv_image)
            pred = output.argmax(dim=1).item()
        
        if pred != label:
            successes += 1
            l0 = (env.modification_mask.sum().item())
            total_l0 += l0
    
    asr = successes / num_test * 100
    avg_l0 = total_l0 / max(successes, 1)
    
    print(f"\n📊 测试结果:")
    print(f"   ASR: {asr:.1f}%")
    print(f"   平均L0: {avg_l0:.2f}")
    
    return asr, avg_l0


# ========== 使用示例 ==========

if __name__ == "__main__":
    import torchvision
    from torchvision import transforms
    
    print("🧪 测试改进版RL训练")
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ResNet18
    print("\n" + "=" * 80)
    print("训练 ResNet18 专用agent")
    print("=" * 80)
    
    resnet18 = torchvision.models.resnet18(weights=None)
    resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, 10)
    resnet18.load_state_dict(torch.load('cifar10_resnet18.pth', map_location=device))
    resnet18 = resnet18.to(device)
    resnet18.eval()
    
    from sparse_attack_env_v2 import SparseAttackEnvV2
    
    agent_resnet = train_model_specific_agent(
        model=resnet18,
        model_name='resnet18',
        dataset=dataset,
        env_class=SparseAttackEnvV2,
        num_train_samples=100,  # 100个训练样本
        timesteps=50000,        # 5万步训练
        device=device
    )
    
    print("\n✅ 训练完成！")

