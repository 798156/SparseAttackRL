# test_multi_sample_training.py
"""
快速测试多样本训练功能
用少量样本和步数验证训练流程是否正常
"""

import torch
import torchvision
from torchvision import transforms
from sparse_attack_env_v2 import SparseAttackEnvV2
from ppo_trainer_v3_improved import DynamicSampleEnv, train_rl_multi_sample
import os

def main():
    print("=" * 80)
    print("🧪 测试多样本训练流程")
    print("=" * 80)
    
    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    if device == 'cpu':
        print("⚠️  警告：使用CPU会很慢")
    
    # 加载ResNet18模型
    print("\n📦 加载ResNet18模型...")
    if not os.path.exists('cifar10_resnet18.pth'):
        print("❌ 错误：ResNet18模型不存在！")
        print("请先确保有 cifar10_resnet18.pth 文件")
        return
    
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load('cifar10_resnet18.pth', map_location=device))
    model = model.to(device)
    model.eval()
    print("✅ 模型加载完成")
    
    # 加载CIFAR-10数据
    print("\n📦 加载CIFAR-10数据...")
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
    print(f"✅ 数据集加载完成，共{len(dataset)}个样本")
    
    # 测试1: 创建动态采样环境
    print("\n" + "=" * 80)
    print("测试1: 创建动态采样环境（10个样本）")
    print("=" * 80)
    
    try:
        def make_env():
            return DynamicSampleEnv(
                env_class=SparseAttackEnvV2,
                model=model,
                dataset=dataset,
                device=device,
                num_samples=10,  # 少量样本测试
                max_steps=5
            )
        
        env = make_env()
        print("✅ 环境创建成功")
        print(f"   动作空间: {env.action_space}")
        print(f"   状态空间: {env.observation_space}")
        
        # 测试环境reset
        print("\n测试环境reset（会随机选择不同样本）...")
        for i in range(3):
            obs, info = env.reset()
            print(f"  Reset {i+1}: obs shape = {obs.shape}")
        print("✅ 环境reset正常")
        
        # 测试环境step
        print("\n测试环境step...")
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step: obs shape = {obs.shape}, reward = {reward:.2f}")
        print("✅ 环境step正常")
        
    except Exception as e:
        print(f"❌ 环境创建失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试2: 快速训练（极少步数）
    print("\n" + "=" * 80)
    print("测试2: 快速训练（1000步，约1-2分钟）")
    print("=" * 80)
    
    try:
        agent = train_rl_multi_sample(
            model=model,
            dataset=dataset,
            env_class=SparseAttackEnvV2,
            num_train_samples=10,  # 10个样本
            timesteps=1000,        # 只训练1000步
            save_path="test_ppo_multi",
            device=device,
            max_steps=5,
            verbose=1
        )
        print("✅ 训练完成")
        
        # 测试训练好的agent
        print("\n测试训练好的agent...")
        env = make_env()
        obs, _ = env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        print(f"  预测动作: {action}")
        print("✅ Agent预测正常")
        
        # 清理测试文件
        if os.path.exists("test_ppo_multi.zip"):
            os.remove("test_ppo_multi.zip")
            print("✅ 清理测试文件")
        
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("🎉 所有测试通过！多样本训练功能正常！")
    print("=" * 80)
    print("\n下一步：开始完整训练（50k步，1-2小时）")


if __name__ == '__main__':
    main()








