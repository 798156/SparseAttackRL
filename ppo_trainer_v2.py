# ppo_trainer_v2.py
"""
优化版本的PPO训练器
主要改进：
1. CNN策略网络（替代MLP）
2. 改进的超参数配置
3. 更好的日志记录
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import gymnasium as gym


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    自定义CNN特征提取器
    用于处理图像输入，保留空间结构
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # 获取输入维度
        n_input_channels = observation_space.shape[0]  # C+3 for enhanced state
        
        # CNN架构
        self.cnn = nn.Sequential(
            # 第一层卷积: (C+3, 32, 32) -> (32, 16, 16)
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二层卷积: (32, 16, 16) -> (64, 8, 8)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层卷积: (64, 8, 8) -> (64, 4, 4)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Flatten
            nn.Flatten(),
        )
        
        # 计算展平后的维度
        # 对于32x32输入: 64 * 4 * 4 = 1024
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, 32, 32)
            n_flatten = self.cnn(sample).shape[1]
        
        # 全连接层
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def train_rl_agent_v2(
    env, 
    timesteps=10000,
    save_path="ppo_sparse_model_v2",
    use_cnn=True,
    learning_rate=3e-4,
    verbose=1
):
    """
    使用优化配置训练 RL 智能体
    
    参数:
        env: 训练环境（最好是 SparseAttackEnvV2）
        timesteps: 训练步数
        save_path: 模型保存路径
        use_cnn: 是否使用CNN策略（推荐True）
        learning_rate: 学习率
        verbose: 日志详细程度
    
    返回:
        model: 训练好的PPO模型
    """
    vec_env = DummyVecEnv([lambda: env])
    
    if use_cnn:
        # 使用CNN策略
        policy_kwargs = dict(
            features_extractor_class=CNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]  # Actor和Critic的隐藏层
        )
        
        model = PPO(
            policy="CnnPolicy",  # 使用CnnPolicy
            env=vec_env,
            learning_rate=learning_rate,
            n_steps=2048,           # 每次更新收集的步数
            batch_size=64,          # 批次大小
            n_epochs=10,            # 每次更新的训练轮数
            gamma=0.99,             # 折扣因子
            gae_lambda=0.95,        # GAE参数
            clip_range=0.2,         # PPO裁剪参数
            ent_coef=0.01,          # 熵系数（鼓励探索）
            vf_coef=0.5,            # 价值函数系数
            max_grad_norm=0.5,      # 梯度裁剪
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log="./logs/",
            device="auto"
        )
        
        print("🧠 使用 CNN 策略网络")
    else:
        # 使用传统MLP策略
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=learning_rate,
            verbose=verbose,
            tensorboard_log="./logs/",
            gamma=0.99,
            ent_coef=0.01,
            batch_size=64,
            device="auto"
        )
        
        print("🧠 使用 MLP 策略网络")
    
    print(f"🚀 开始训练 RL 智能体，共 {timesteps} 步...")
    print(f"   学习率: {learning_rate}")
    print(f"   策略类型: {'CNN' if use_cnn else 'MLP'}")
    
    model.learn(total_timesteps=timesteps, tb_log_name="ppo_v2_run")
    model.save(save_path)
    
    print(f"💾 模型已保存至: {save_path}.zip")
    
    return model


def train_with_curriculum(
    env_template,
    test_set,
    model,
    timesteps_per_stage=3000,
    save_path="ppo_curriculum"
):
    """
    课程学习训练
    从简单样本逐步过渡到困难样本
    
    参数:
        env_template: 环境创建函数
        test_set: 测试数据集
        model: 目标模型
        timesteps_per_stage: 每个阶段的训练步数
        save_path: 模型保存路径
    
    返回:
        agent: 训练好的智能体
    """
    print("\n" + "=" * 60)
    print("📚 开始课程学习训练")
    print("=" * 60)
    
    # 根据置信度对样本进行分级
    print("\n📊 评估样本难度...")
    difficulties = []
    
    for i in range(min(1000, len(test_set))):
        image, label = test_set[i]
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(next(model.parameters()).device))
            pred = output.argmax(dim=1).item()
            conf = torch.softmax(output, dim=1)[0, label].item()
        
        if pred == label:  # 只考虑正确分类的样本
            difficulties.append((i, conf))
    
    # 按置信度排序
    difficulties.sort(key=lambda x: x[1])
    
    # 分为三个难度等级
    n_samples = len(difficulties)
    easy_samples = [idx for idx, _ in difficulties[:n_samples//3]]
    medium_samples = [idx for idx, _ in difficulties[n_samples//3:2*n_samples//3]]
    hard_samples = [idx for idx, _ in difficulties[2*n_samples//3:]]
    
    print(f"   简单样本: {len(easy_samples)} 个")
    print(f"   中等样本: {len(medium_samples)} 个")
    print(f"   困难样本: {len(hard_samples)} 个")
    
    # 阶段1: 简单样本
    print("\n📖 阶段1: 训练简单样本（置信度低）")
    sample_idx = easy_samples[0]
    image, label = test_set[sample_idx]
    env = env_template(image, label, model, max_steps=7)
    
    agent = train_rl_agent_v2(
        env, 
        timesteps=timesteps_per_stage,
        save_path=f"{save_path}_stage1",
        use_cnn=True
    )
    
    # 阶段2: 中等样本
    print("\n📖 阶段2: 训练中等样本")
    sample_idx = medium_samples[0]
    image, label = test_set[sample_idx]
    env = env_template(image, label, model, max_steps=5)
    
    agent.set_env(DummyVecEnv([lambda: env]))
    agent.learn(total_timesteps=timesteps_per_stage, tb_log_name="curriculum_stage2", reset_num_timesteps=False)
    agent.save(f"{save_path}_stage2")
    
    # 阶段3: 困难样本
    print("\n📖 阶段3: 训练困难样本（置信度高）")
    sample_idx = hard_samples[0]
    image, label = test_set[sample_idx]
    env = env_template(image, label, model, max_steps=3)
    
    agent.set_env(DummyVecEnv([lambda: env]))
    agent.learn(total_timesteps=timesteps_per_stage, tb_log_name="curriculum_stage3", reset_num_timesteps=False)
    agent.save(f"{save_path}_final")
    
    print("\n✅ 课程学习训练完成！")
    
    return agent


# 使用示例
if __name__ == "__main__":
    print("🧪 测试 PPO Trainer V2")
    
    from torchvision import datasets, transforms
    from target_model import load_target_model
    from sparse_attack_env_v2 import SparseAttackEnvV2
    
    # 加载数据和模型
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    target_model = load_target_model('resnet18', num_classes=10)
    
    image, label = test_set[0]
    
    # 创建优化版环境
    env = SparseAttackEnvV2(image, label, target_model, max_steps=5, use_saliency=True)
    
    # 训练（少量步数用于测试）
    print("\n开始训练...")
    agent = train_rl_agent_v2(
        env, 
        timesteps=1000,  # 测试用，实际应该用10000+
        save_path="test_ppo_v2",
        use_cnn=True
    )
    
    print("\n✅ 测试完成！")

