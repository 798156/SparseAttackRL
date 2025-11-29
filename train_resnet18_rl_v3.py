# train_resnet18_rl_v3.py
"""
为ResNet18训练改进的RL agent (V3)
使用多样本训练，预期ASR达到70%+
"""

import torch
import torchvision
from torchvision import transforms
from sparse_attack_env_v2 import SparseAttackEnvV2
from ppo_trainer_v3_improved import train_rl_multi_sample, train_model_specific_agent
import os
import time

def main():
    print("=" * 80)
    print("🎯 ResNet18 RL V3 完整训练")
    print("=" * 80)
    
    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    if device == 'cpu':
        print("⚠️  警告：使用CPU训练会非常慢！")
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            return
    
    # 加载ResNet18模型
    print("\n📦 加载ResNet18模型...")
    if not os.path.exists('cifar10_resnet18.pth'):
        print("❌ 错误：ResNet18模型不存在！")
        print("请先训练ResNet18模型或使用现有的模型")
        return
    
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load('cifar10_resnet18.pth', 
                                     map_location=device, 
                                     weights_only=False))
    model = model.to(device)
    model.eval()
    
    # 验证模型准确率
    print("\n🔍 验证模型准确率...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, 
                                             shuffle=False, num_workers=0)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    print(f"✅ 模型准确率: {accuracy:.2f}%")
    
    if accuracy < 75:
        print(f"⚠️  警告：模型准确率较低（{accuracy:.2f}%），可能影响训练效果")
    
    # 训练配置
    print("\n" + "=" * 80)
    print("🎓 训练配置")
    print("=" * 80)
    
    config = {
        'num_samples': 100,      # 训练样本数
        'timesteps': 50000,      # 训练步数
        'max_steps': 5,          # 每个episode最大步数
        'learning_rate': 3e-4,   # 学习率
        'save_path': 'models/ppo_resnet18_v3'
    }
    
    print(f"  训练样本数: {config['num_samples']}")
    print(f"  训练步数: {config['timesteps']:,}")
    print(f"  最大步数: {config['max_steps']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  保存路径: {config['save_path']}")
    print(f"\n预计训练时间: 1-2小时（GPU）")
    
    # 确认
    print("\n" + "=" * 80)
    response = input("开始训练？(y/n): ")
    if response.lower() != 'y':
        print("训练取消")
        return
    
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    
    # 开始训练
    start_time = time.time()
    
    try:
        agent = train_rl_multi_sample(
            model=model,
            dataset=testset,
            env_class=SparseAttackEnvV2,
            num_train_samples=config['num_samples'],
            timesteps=config['timesteps'],
            save_path=config['save_path'],
            device=device,
            max_steps=config['max_steps'],
            verbose=1
        )
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("🎉 训练完成！")
        print("=" * 80)
        print(f"⏱️  训练耗时: {elapsed/60:.1f}分钟")
        print(f"📁 模型已保存: {config['save_path']}.zip")
        
        # 快速验证
        print("\n" + "=" * 80)
        print("🧪 快速验证（10个样本）")
        print("=" * 80)
        
        successes = 0
        total_l0 = 0
        
        for i in range(10):
            image, label = testset[i]
            
            # 检查是否正确分类
            with torch.no_grad():
                output = model(image.unsqueeze(0).to(device))
                pred = output.argmax(dim=1).item()
            
            if pred != label:
                continue
            
            # 创建环境并攻击
            env = SparseAttackEnvV2(
                clean_image=image,
                true_label=label,
                model=model,
                max_steps=5,
                use_saliency=True
            )
            
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
                output = model(env.current_image)
                pred = output.argmax(dim=1).item()
            
            if pred != label:
                successes += 1
                total_l0 += env.current_step
                print(f"  样本{i}: ✅ 成功 (L0={env.current_step})")
            else:
                print(f"  样本{i}: ❌ 失败")
        
        asr = successes / 10 * 100
        avg_l0 = total_l0 / max(successes, 1)
        
        print(f"\n快速验证结果:")
        print(f"  ASR: {asr:.0f}%")
        print(f"  平均L0: {avg_l0:.2f}")
        
        if asr >= 70:
            print("\n🎉 太好了！ASR达到70%+！")
        elif asr >= 60:
            print("\n✅ 不错！ASR达到60%+")
        else:
            print("\n⚠️  ASR较低，可能需要继续训练或调整参数")
        
        print("\n" + "=" * 80)
        print("下一步：运行完整测试（100样本）")
        print("命令：python test_resnet18_rl_v3.py")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被中断")
        elapsed = time.time() - start_time
        print(f"已训练: {elapsed/60:.1f}分钟")
        
        # 尝试保存当前模型
        try:
            if 'agent' in locals():
                interrupted_path = config['save_path'] + '_interrupted'
                agent.save(interrupted_path)
                print(f"💾 当前模型已保存: {interrupted_path}.zip")
        except:
            pass
    
    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

