# diagnose_rl_training.py
"""
诊断RL训练问题
检查环境、agent、训练过程是否正常
"""

import torch
import torchvision
from torchvision import transforms
from sparse_attack_env_v2 import SparseAttackEnvV2
from stable_baselines3 import PPO
import numpy as np

def main():
    print("=" * 80)
    print("🔍 诊断RL训练问题")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型
    print("\n1️⃣ 加载ResNet18模型...")
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load('cifar10_resnet18.pth', 
                                     map_location=device, 
                                     weights_only=False))
    model = model.to(device)
    model.eval()
    print("✅ 模型加载成功")
    
    # 加载数据
    print("\n2️⃣ 加载测试数据...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 选择一个正确分类的样本
    for i in range(100):
        image, label = dataset[i]
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
        
        if pred == label:
            test_image = image
            test_label = label
            print(f"✅ 选择样本{i}，标签{label}")
            break
    
    # 测试环境
    print("\n3️⃣ 测试环境...")
    env = SparseAttackEnvV2(
        clean_image=test_image,
        true_label=test_label,
        model=model,
        max_steps=5,
        use_saliency=True
    )
    
    obs, _ = env.reset()
    print(f"  观测空间: {obs.shape}")
    print(f"  动作空间: {env.action_space}")
    
    # 测试随机动作
    print("\n4️⃣ 测试随机动作...")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {step+1}: reward={reward:.3f}, done={terminated or truncated}")
    
    # 检查最终状态
    with torch.no_grad():
        output = model(env.current_image)
        pred = output.argmax(dim=1).item()
        conf = torch.softmax(output, dim=1)[0, test_label].item()
    
    print(f"  原始标签: {test_label}")
    print(f"  最终预测: {pred}")
    print(f"  置信度: {conf:.3f}")
    print(f"  修改步数: {env.current_step}")
    
    # 加载训练好的agent
    print("\n5️⃣ 加载训练好的RL agent...")
    try:
        agent = PPO.load('models/ppo_resnet18_v3', device=device)
        print("✅ Agent加载成功")
    except Exception as e:
        print(f"❌ Agent加载失败: {e}")
        return
    
    # 测试agent
    print("\n6️⃣ 测试Agent攻击...")
    env = SparseAttackEnvV2(
        clean_image=test_image,
        true_label=test_label,
        model=model,
        max_steps=5,
        use_saliency=True
    )
    
    obs, _ = env.reset()
    
    for step in range(10):
        action, _ = agent.predict(obs, deterministic=True)
        print(f"  Step {step+1}: action={action}")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"           reward={reward:.3f}, done={terminated or truncated}")
        
        if terminated or truncated:
            break
    
    # 检查结果
    with torch.no_grad():
        output = model(env.current_image)
        pred = output.argmax(dim=1).item()
        conf = torch.softmax(output, dim=1)[0, test_label].item()
    
    print(f"\n攻击结果:")
    print(f"  原始标签: {test_label}")
    print(f"  最终预测: {pred}")
    print(f"  置信度: {conf:.3f}")
    print(f"  修改步数: {env.current_step}")
    print(f"  成功: {pred != test_label}")
    
    # 检查agent的输出分布
    print("\n7️⃣ 检查Agent输出分布...")
    env = SparseAttackEnvV2(
        clean_image=test_image,
        true_label=test_label,
        model=model,
        max_steps=5,
        use_saliency=True
    )
    obs, _ = env.reset()
    
    actions = []
    for _ in range(20):
        action, _ = agent.predict(obs, deterministic=False)
        actions.append(action)
    
    actions = np.array(actions)
    print(f"  动作样本数: {len(actions)}")
    print(f"  X 位置范围: {actions[:, 0].min():.1f} - {actions[:, 0].max():.1f}")
    print(f"  Y 位置范围: {actions[:, 1].min():.1f} - {actions[:, 1].max():.1f}")
    print(f"  R 扰动范围: {actions[:, 2].min():.3f} - {actions[:, 2].max():.3f}")
    print(f"  G 扰动范围: {actions[:, 3].min():.3f} - {actions[:, 3].max():.3f}")
    print(f"  B 扰动范围: {actions[:, 4].min():.3f} - {actions[:, 4].max():.3f}")
    
    # 问题诊断
    print("\n" + "=" * 80)
    print("🔧 问题诊断")
    print("=" * 80)
    
    if actions[:, 2:].max() < 0.01:
        print("⚠️  问题1: RGB扰动太小！Agent没有学会大幅修改像素")
    
    if len(set(actions[:, 0])) < 5:
        print("⚠️  问题2: X位置多样性低！Agent总是修改相同位置")
    
    if len(set(actions[:, 1])) < 5:
        print("⚠️  问题3: Y位置多样性低！Agent总是修改相同位置")
    
    print("\n可能的原因:")
    print("1. 奖励信号太弱，agent没有学到有效策略")
    print("2. 训练样本太难，agent放弃学习")
    print("3. max_steps太小（5步），限制了agent")
    print("4. 环境配置有问题")
    
    print("\n建议的解决方案:")
    print("1. 增加max_steps: 5 → 10")
    print("2. 调整奖励函数权重")
    print("3. 使用更简单的样本（低置信度）")
    print("4. 增加训练步数: 50k → 100k")


if __name__ == '__main__':
    main()








