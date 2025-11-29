# train_resnet18_rl_v3_fixed.py
"""
修复版：ResNet18 RL V3训练
关键改进：
1. 增加max_steps: 5 → 10
2. 选择简单样本（低置信度）
3. 增加置信度奖励权重
4. 延长训练时间
"""

import torch
import torchvision
from torchvision import transforms
from sparse_attack_env_v2 import SparseAttackEnvV2
import numpy as np
import os
import time

# 修改环境，增加奖励权重
class EnhancedSparseAttackEnvV2(SparseAttackEnvV2):
    """增强版环境：更强的奖励信号"""
    
    def __init__(self, clean_image, true_label, model, max_steps=10, 
                 use_saliency=True, confidence_reward_weight=15.0):
        # 使用更高的置信度奖励权重
        super().__init__(clean_image, true_label, model, max_steps, 
                        use_saliency, confidence_reward_weight)
        
        self.last_modified_pos = None
    
    def step(self, action):
        """增强版step：惩罚重复修改同一位置"""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 获取修改位置
        x, y = int(action[0]), int(action[1])
        current_pos = (x, y)
        
        # 如果连续修改同一位置，额外惩罚
        if self.last_modified_pos == current_pos and not (terminated or truncated):
            reward -= 0.5  # 额外惩罚
        
        self.last_modified_pos = current_pos
        
        return obs, reward, terminated, truncated, info


def select_easy_samples(model, dataset, num_samples=100, device='cuda'):
    """选择简单样本（低置信度）"""
    print(f"📊 选择{num_samples}个简单样本（置信度<0.85）...")
    
    sample_difficulties = []
    
    with torch.no_grad():
        for idx in range(min(len(dataset), num_samples * 5)):
            image, label = dataset[idx]
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            pred = output.argmax(dim=1).item()
            
            if pred == label:
                conf = torch.softmax(output, dim=1)[0, label].item()
                # 只选择置信度<0.85的样本（更容易攻击）
                if conf < 0.85:
                    sample_difficulties.append((idx, conf))
            
            if len(sample_difficulties) >= num_samples:
                break
    
    # 按置信度排序，先训练简单的
    sample_difficulties.sort(key=lambda x: x[1])
    selected = [idx for idx, _ in sample_difficulties[:num_samples]]
    
    if selected:
        avg_conf = np.mean([conf for _, conf in sample_difficulties[:num_samples]])
        print(f"✅ 选择了{len(selected)}个样本，平均置信度: {avg_conf:.3f}")
    else:
        print("⚠️  没找到足够的简单样本，使用所有正确分类的样本")
        selected = []
        for idx in range(min(len(dataset), num_samples * 2)):
            image, label = dataset[idx]
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            pred = output.argmax(dim=1).item()
            if pred == label:
                selected.append(idx)
            if len(selected) >= num_samples:
                break
    
    return selected


def main():
    print("=" * 80)
    print("🎯 ResNet18 RL V3 修复版训练")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 加载模型
    print("\n📦 加载ResNet18模型...")
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load('cifar10_resnet18.pth', 
                                     map_location=device, 
                                     weights_only=False))
    model = model.to(device)
    model.eval()
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 选择简单样本
    easy_samples = select_easy_samples(model, dataset, num_samples=100, device=device)
    
    if len(easy_samples) < 50:
        print("❌ 简单样本太少，无法训练")
        return
    
    # 训练配置
    print("\n" + "=" * 80)
    print("🎓 改进的训练配置")
    print("=" * 80)
    
    config = {
        'max_steps': 10,           # 增加！5→10
        'timesteps': 80000,        # 增加！50k→80k
        'confidence_weight': 15.0, # 增加！5→15
        'save_path': 'models/ppo_resnet18_v3_fixed'
    }
    
    print(f"  训练样本数: {len(easy_samples)} (简单样本)")
    print(f"  max_steps: {config['max_steps']} (原来5)")
    print(f"  训练步数: {config['timesteps']:,} (原来50k)")
    print(f"  置信度奖励权重: {config['confidence_weight']} (原来5.0)")
    print(f"\n预计训练时间: 1.5-2.5小时（GPU）")
    
    response = input("\n开始训练？(y/n): ")
    if response.lower() != 'y':
        return
    
    # 使用改进的训练方法
    from ppo_trainer_v3_improved import DynamicSampleEnv, train_rl_multi_sample
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3 import PPO
    from ppo_trainer_v2 import CNNFeatureExtractor
    
    # 创建环境包装器（使用增强版环境）
    class EasyDynamicEnv:
        def __init__(self):
            self.easy_samples = easy_samples
            self.dataset = dataset
            self.model = model
            self.device = device
            self.max_steps = config['max_steps']
            self.current_env = None
            self._reset_with_new_sample()
            self.action_space = self.current_env.action_space
            self.observation_space = self.current_env.observation_space
        
        def _reset_with_new_sample(self):
            idx = np.random.choice(self.easy_samples)
            image, label = self.dataset[idx]
            self.current_env = EnhancedSparseAttackEnvV2(
                clean_image=image,
                true_label=label,
                model=self.model,
                max_steps=self.max_steps,
                use_saliency=True,
                confidence_reward_weight=config['confidence_weight']
            )
        
        def reset(self, **kwargs):
            self._reset_with_new_sample()
            obs, info = self.current_env.reset(**kwargs)
            return obs, info  # 必须返回元组
        
        def step(self, action):
            return self.current_env.step(action)
    
    # 导入gym以继承
    import gymnasium as gym
    
    # 使gym能识别
    class EasyDynamicEnvGym(gym.Env, EasyDynamicEnv):
        def __init__(self):
            gym.Env.__init__(self)
            EasyDynamicEnv.__init__(self)
    
    # 创建环境
    def make_env():
        return EasyDynamicEnvGym()
    
    env = DummyVecEnv([make_env])
    
    # 创建agent
    policy_kwargs = dict(
        features_extractor_class=CNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[128, 128], vf=[128, 128])]
    )
    
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
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./logs/",
        device=device
    )
    
    # 训练
    print("\n🚀 开始训练...")
    start_time = time.time()
    
    try:
        agent.learn(
            total_timesteps=config['timesteps'],
            tb_log_name="ppo_resnet18_v3_fixed",
            progress_bar=True
        )
        
        elapsed = time.time() - start_time
        
        # 保存
        os.makedirs('models', exist_ok=True)
        agent.save(config['save_path'])
        
        print("\n" + "=" * 80)
        print("🎉 训练完成！")
        print(f"⏱️  耗时: {elapsed/60:.1f}分钟")
        print(f"📁 保存: {config['save_path']}.zip")
        print("=" * 80)
        
        # 快速验证
        print("\n🧪 快速验证（10个样本）...")
        successes = 0
        
        for i in range(10):
            idx = easy_samples[i]
            image, label = dataset[idx]
            
            env_test = EnhancedSparseAttackEnvV2(
                clean_image=image,
                true_label=label,
                model=model,
                max_steps=config['max_steps'],
                use_saliency=True
            )
            
            obs, _ = env_test.reset()
            done = False
            steps = 0
            
            while not done and steps < 15:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env_test.step(action)
                done = terminated or truncated
                steps += 1
            
            with torch.no_grad():
                output = model(env_test.current_image)
                pred = output.argmax(dim=1).item()
            
            if pred != label:
                successes += 1
                print(f"  样本{i}: ✅ 成功")
            else:
                print(f"  样本{i}: ❌ 失败")
        
        asr = successes / 10 * 100
        print(f"\nASR: {asr:.0f}%")
        
        if asr >= 60:
            print("🎉 成功！ASR达标！")
        else:
            print("⚠️  ASR仍然较低，可能需要进一步调整")
            
    except KeyboardInterrupt:
        print("\n训练中断")
        agent.save(config['save_path'] + '_interrupted')
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

