# train_rl_v3_simple.py
"""
简化版RL V3训练 - 直接使用已测试的代码
关键改进：
1. max_steps增加到10
2. 选择简单样本（低置信度）
3. 增加置信度奖励权重到15
"""

import torch
import torchvision
from torchvision import transforms
from sparse_attack_env_v2 import SparseAttackEnvV2
import numpy as np
import os
import time


def select_easy_samples(model, dataset, num_samples=100, max_conf=0.85, device='cuda'):
    """选择简单样本（低置信度）"""
    print(f"📊 选择简单样本（置信度<{max_conf}）...")
    
    easy_indices = []
    confidences = []
    
    with torch.no_grad():
        for idx in range(min(len(dataset), num_samples * 5)):
            image, label = dataset[idx]
            output = model(image.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
            
            if pred == label:
                conf = torch.softmax(output, dim=1)[0, label].item()
                if conf < max_conf:
                    easy_indices.append(idx)
                    confidences.append(conf)
            
            if len(easy_indices) >= num_samples:
                break
    
    if easy_indices:
        avg_conf = np.mean(confidences)
        print(f"✅ 选择了{len(easy_indices)}个样本，平均置信度: {avg_conf:.3f}")
    else:
        print("⚠️  找不到足够简单样本，使用所有正确分类的样本")
        for idx in range(min(len(dataset), num_samples * 2)):
            image, label = dataset[idx]
            output = model(image.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
            if pred == label:
                easy_indices.append(idx)
            if len(easy_indices) >= num_samples:
                break
    
    return easy_indices


def main():
    print("=" * 80)
    print("🎯 RL V3 简化版训练")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")
    
    # 加载模型
    print("📦 加载ResNet18...")
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load('cifar10_resnet18.pth', 
                                     map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                          download=True, transform=transform)
    
    # 选择简单样本
    easy_samples = select_easy_samples(model, dataset, num_samples=100, device=device)
    
    if len(easy_samples) < 30:
        print("❌ 简单样本太少")
        return
    
    # 配置
    print("\n" + "=" * 80)
    print("配置:")
    print(f"  训练样本: {len(easy_samples)}")
    print(f"  max_steps: 10")
    print(f"  timesteps: 80,000")
    print(f"  置信度权重: 15.0")
    print("=" * 80)
    
    response = input("\n开始训练？(y/n): ")
    if response.lower() != 'y':
        return
    
    # 使用已测试的DynamicSampleEnv
    from ppo_trainer_v3_improved import DynamicSampleEnv
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3 import PPO
    from ppo_trainer_v2 import CNNFeatureExtractor
    import gymnasium as gym
    
    # 创建修改过的环境类
    class EasySparseAttackEnvV2(SparseAttackEnvV2):
        """增强版：更强的置信度奖励"""
        def __init__(self, clean_image, true_label, model, max_steps=10):
            super().__init__(clean_image, true_label, model, max_steps, 
                           use_saliency=True, confidence_reward_weight=15.0)
    
    # 创建动态环境
    class EasyDynamicSampleEnv(gym.Env):
        def __init__(self):
            super().__init__()
            idx = np.random.choice(easy_samples)
            image, label = dataset[idx]
            self.current_env = EasySparseAttackEnvV2(image, label, model)
            self.action_space = self.current_env.action_space
            self.observation_space = self.current_env.observation_space
        
        def reset(self, **kwargs):
            # 随机选择新样本
            idx = np.random.choice(easy_samples)
            image, label = dataset[idx]
            self.current_env = EasySparseAttackEnvV2(image, label, model)
            return self.current_env.reset(**kwargs)
        
        def step(self, action):
            return self.current_env.step(action)
    
    # 创建向量化环境
    env = DummyVecEnv([lambda: EasyDynamicSampleEnv()])
    
    # 创建PPO agent
    policy_kwargs = dict(
        features_extractor_class=CNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 128], vf=[128, 128])  # 修正格式
    )
    
    agent = PPO(
        "CnnPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./logs/",
        device=device
    )
    
    # 训练
    print("\n🚀 开始训练...\n")
    start_time = time.time()
    
    try:
        agent.learn(total_timesteps=80000, progress_bar=True,
                   tb_log_name="ppo_resnet18_v3_simple")
        
        # 保存
        os.makedirs('models', exist_ok=True)
        agent.save('models/ppo_resnet18_v3_simple')
        
        elapsed = time.time() - start_time
        print(f"\n🎉 训练完成！耗时: {elapsed/60:.1f}分钟")
        print("📁 保存: models/ppo_resnet18_v3_simple.zip")
        
        # 快速验证
        print("\n🧪 快速验证...")
        successes = 0
        
        for i in range(10):
            idx = easy_samples[i]
            image, label = dataset[idx]
            
            test_env = EasySparseAttackEnvV2(image, label, model)
            obs, _ = test_env.reset()
            
            for _ in range(15):
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                if terminated or truncated:
                    break
            
            with torch.no_grad():
                pred = model(test_env.current_image).argmax(dim=1).item()
            
            if pred != label:
                successes += 1
                print(f"  样本{i}: ✅")
            else:
                print(f"  样本{i}: ❌")
        
        print(f"\nASR: {successes*10}%")
        
    except KeyboardInterrupt:
        print("\n⚠️  训练中断")
        agent.save('models/ppo_resnet18_v3_simple_interrupted')
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()








